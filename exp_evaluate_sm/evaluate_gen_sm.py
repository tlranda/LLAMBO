import os
import argparse
import json
import optuna
import logging
import pickle
import numpy as np
import pandas as pd
from bayesmark.bbox_utils import get_bayesmark_func
from llambo.rate_limiter import RateLimiter
from llambo.generative_sm import LLM_GEN_SM
from exp_evaluate_sm.evaluate_sm_utils import fit_and_predict_with_TPE
from sklearn.metrics import get_scorer
from sklearn.metrics import balanced_accuracy_score
# switch off future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

rate_limiter = RateLimiter(max_tokens=240000, time_frame=60, max_requests=2900)

logger = logging.getLogger(__name__)

def setup_logging(log_name):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

def load_precomputed_samples(hp_constraints, n, dataset_name, model, dataset):
    # For purposes here, the train/test split does not matter, re-unify them
    # Ignore the n-value for now, we may revisit that later
    all_x = np.vstack((dataset['train_x'], dataset['test_x']))
    all_y = np.hstack((dataset['train_y'], dataset['test_y']))
    sampled_configs = pd.DataFrame(all_x, columns=list(hp_constraints.keys()))
    sampled_scores = pd.DataFrame(all_y, columns=['score'])
    return sampled_configs, sampled_scores

def obtain_n_configurations(hp_constraints, n, dataset, model, task_metric, task_type, lower_is_better):
    # run random sampled hyperaparameter configurations with optuna
    def objective(trial):
        config = {}
        for hp_name, hp_info in hp_constraints.items():
            use_log = hp_info[1] in ['log', 'logit']
            if hp_info[0] == 'int':
                config[hp_name] = trial.suggest_int(hp_name, hp_info[2], hp_info[3], log=use_log)
            elif hp_info[0] == 'float':
                config[hp_name] = trial.suggest_float(hp_name, hp_info[2], hp_info[3], log=use_log)
            else:
                raise ValueError(f'Unknown hyperparameter type: {hp_info[0]}')
            
        model_ = get_bayesmark_func(model, task_type)

        train_x = dataset['train_x']
        test_x = dataset['test_x']
        
        if task_type == 'regression':
            # standardize y
            y_mean = dataset['train_y'].mean()
            y_std = dataset['train_y'].std()
            train_y = (dataset['train_y'] - y_mean) / y_std
            test_y = (dataset['test_y'] - y_mean) / y_std
        else:
            train_y = dataset['train_y']
            test_y = dataset['test_y']

        if model == 'SVM':
            predictor = model_(**config)
        else:
            predictor = model_(**config, random_state=42)
        predictor.fit(train_x, train_y)
        scorer = get_scorer(task_metric)
        score = scorer(predictor, test_x, test_y)
        return score
    
    configs = []
    scores = []
    for i in range(5):
        direction = 'minimize' if lower_is_better else 'maximize'
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42+i), direction=direction)
        study.optimize(objective, n_trials=n)

        # get all configurations and scores
        for trial in study.trials:
            configs.append(trial.params)
            if task_metric == 'neg_mean_squared_error':
                scores.append(-trial.value)
            else:
                scores.append(trial.value)


    configs = pd.DataFrame(configs)
    scores = pd.DataFrame(scores, columns=['score'])

    return configs, scores


def sample_n_configurations(configs, scores, n, seed, existing_config=None, local_quantiles=False):
    '''Sample n configurations from configs and scores'''
    number_sampled = 0
    iter_i = 0

    if local_quantiles:
        score_quantiles = scores[['score']].rank(method='max').apply(lambda x: (x-1)/(len(scores)-1))

    sampled_configs = pd.DataFrame()
    sampled_scores = pd.DataFrame()
    
    # get all unique values in scores
    unique_scores = scores['score'].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_scores)

    scores['score_rank'] = scores['score'].apply(lambda x: np.where(unique_scores == x)[0][0])

    # avoid execessive duplication of configs - makes prediction task trivial!
    while number_sampled < n:
        if local_quantiles:
            # randomly sample from uniform quantiles, but account for loop iteration in case existing_config denies certain values
            sample_index = np.array([np.floor(len(scores)*(_/n)) for _ in range(n)], dtype=int) + iter_i
            # If things get dropped due to existing_config, you may see less-uniform overall distribution, so shuffle the indices to counteract
            np.random.shuffle(sample_index)
        else:
            # randomly sample from each unique score
            sample_index = scores.groupby('score_rank').apply(lambda x: x.sample(1, random_state=seed+iter_i)).index.get_level_values(1)
        # get sampled configs and scores
        sampled_configs = pd.concat([sampled_configs, configs.iloc[sample_index]], axis=0)
        if local_quantiles:
            sampled_scores = pd.concat([sampled_scores, score_quantiles.iloc[sample_index]], axis=0)
        else:
            sampled_scores = pd.concat([sampled_scores, scores[['score']].iloc[sample_index]], axis=0)
        sampled_configs = sampled_configs.reset_index(drop=True)
        sampled_scores = sampled_scores.reset_index(drop=True)

        if existing_config is not None:
            drop_index = []
            for i in range(sampled_configs.shape[0]):
                row = sampled_configs.iloc[i, :]
                if (existing_config == row).all(1).any():
                    drop_index.append(i)
            
            sampled_configs = sampled_configs.drop(drop_index)
            sampled_scores = sampled_scores.drop(drop_index)
            sampled_configs = sampled_configs.reset_index(drop=True)
            sampled_scores = sampled_scores.reset_index(drop=True)

        # remove duplicates
        duplicate_index = sampled_configs[sampled_configs.duplicated()].index
        sampled_configs = sampled_configs.drop(duplicate_index)
        sampled_scores = sampled_scores.drop(duplicate_index)
        sampled_configs = sampled_configs.reset_index(drop=True)
        sampled_scores = sampled_scores.reset_index(drop=True)

        iter_i += 1
        number_sampled = len(sampled_configs)

    sampled_configs = sampled_configs.head(n)
    sampled_scores = sampled_scores.head(n)

    return sampled_configs, sampled_scores

def evaluate_score(pred_score, candidate_fvals, observed_fvals, lower_is_better, top_pct):
    candidate_fvals = candidate_fvals.copy().values

    candidate_fvals = candidate_fvals.squeeze()

    if lower_is_better:
        candidate_fvals = -candidate_fvals

    # calculate correlation
    corr = np.corrcoef(pred_score, candidate_fvals)[0, 1]

    # calculate regret
    index = np.argmax(pred_score)
    if lower_is_better:
        regret = candidate_fvals[index] - candidate_fvals.min()
    else:
        regret = candidate_fvals.max() - candidate_fvals[index]
    # normalize regret
    regret /= np.abs(candidate_fvals.max() - np.abs(candidate_fvals).min())
    print(f'Point acquired: {index}')

    return corr, regret

TASK_MAP = {
    'breast': ['classification', 'accuracy'],
    'digits': ['classification', 'accuracy'],
    'wine': ['classification', 'accuracy'],
    'iris': ['classification', 'accuracy'],
    'diabetes': ['regression', 'neg_mean_squared_error'],
    'syr2k': ['regression', 'neg_mean_squared_error'],
    'syr2k_q': ['quantile-prediction', 'quantile'],
    'syr2k_r': ['rank-prediction', 'rank out of 10648'],
}

if __name__ == '__main__':
    # load hyperparameter config space
    with open(f'hp_configurations/bayesmark.json', 'r') as f:
        hp_constraints = json.load(f)
    parser = argparse.ArgumentParser()
    # Extensions:
    # * DatasetIdentity:syr2k (tuning options for syr2k)
    parser.add_argument('--model', type=str, choices=list(hp_constraints.keys()), help="Tunable parameters the LLM is asked to reason about")
    # Extensions:
    # * syr2k (syr2k data with actual objective data)
    # * syr2k_q (syr2k data but with quantiles as objectives instead of actual objective)
    # * syr2k_r (syr2k data but with ranks as objectives instead of actual objective)
    parser.add_argument('--dataset', type=str, choices=list(TASK_MAP.keys()), help="Data from interacting with the model")
    parser.add_argument('--num_observed', type=int, help="Number of ICL examples for the LLM")
    parser.add_argument('--num_seeds', type=int, default=1, help="Number of LLM seeds to try (default: %(default)s)")
    parser.add_argument('--num_candidates', type=int, default=10, help="Number of candidates for LLM to try (default: %(default)s)")
    parser.add_argument('--engine', type=str, default='llama3', help="LLM model to use for inference (default: %(default)s)")
    parser.add_argument('--evaluate', nargs='+', required=True, default=None, choices=['TPE_Multi','TPE_Ind','LLAMBO','LLAMBO_VANILLA'], help="Techniques to use during evaluation")

    args = parser.parse_args()
    model = args.model
    dataset = args.dataset
    num_observed = args.num_observed
    num_candidates = args.num_candidates
    num_seeds = args.num_seeds
    engine = args.engine
    to_evaluate = args.evaluate

    hp_constraints = hp_constraints[model]
    task_map = TASK_MAP[dataset]
    task_type = task_map[0]
    task_metric = task_map[1]
    # define result save directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_res_fpath = f'{script_dir}/results/evaluate_gen_sm/{dataset}/{model}/{num_observed}.json'
    if not os.path.exists(os.path.dirname(save_res_fpath)):
        os.makedirs(os.path.dirname(save_res_fpath))
    # define logging directory
    logging_fpath = f'{script_dir}/logs/evaluate_gen_sm/{dataset}/{model}/{num_observed}.log'
    if not os.path.exists(os.path.dirname(logging_fpath)):
        os.makedirs(os.path.dirname(logging_fpath))
    setup_logging(logging_fpath)

    logger.info('='*200)
    logger.info(f'Evaluating Generative SM performance on {dataset} with {model} and {num_observed} observed configurations... Running {num_seeds} runs.')
    logger.info('='*200)

    # load dataset
    pickle_fpath = f'bayesmark/data/{dataset}.pickle'
    with open(pickle_fpath, 'rb') as f:
        dataset_loaded = pickle.load(f)

    results = {}
    for evaluator in to_evaluate:
        results[evaluator] = dict((key, list()) for key in ['corr','regret',
                                                            'y_prob','y_true'])
        if 'LLAMBO' in evaluator:
            results[evaluator].update(dict((key, list()) for key in ['llm_query_cost','llm_query_time']))
    lower_is_better = False if task_metric == 'accuracy' else True

    logger.info(f'Collecting configurations - this might take a while...')
    if model.startswith('DatasetIdentity:'):
        sampled_configs, sampled_scores = load_precomputed_samples(hp_constraints, 100, dataset_loaded, model, dataset_loaded)
    else:
        sampled_configs, sampled_scores = obtain_n_configurations(hp_constraints, 100, dataset_loaded, model, 
                                                          task_metric=task_metric, task_type=task_type, lower_is_better=lower_is_better)
    logger.info("Sampled Configurations")
    logger.info(sampled_configs)
    logger.info("Sampled Scores")
    logger.info(sampled_scores)

    tot_llm_cost = 0
    for seed in range(num_seeds):
        logger.info('='*200)
        logger.info(f'Evaluating SM with seed {seed}...')

        observed_configs, observed_fvals = sample_n_configurations(sampled_configs, sampled_scores, num_observed, seed=seed, local_quantiles=dataset.endswith('_q'))
        logger.info("Observed_configs (ICL)")
        logger.info(observed_configs)
        logger.info("Observed_fvals (ICL)")
        logger.info(observed_fvals)
        logger.info('.'*50)
        # LLAMBO authors forgot to pass in the existing config argument to prevent LLM from being evaluated against ICL data
        candidate_configs, candidate_fvals = sample_n_configurations(sampled_configs, sampled_scores, num_candidates, seed=42, existing_config=observed_configs, local_quantiles=dataset.endswith('_q'))
        logger.info("Prompt configs to solve via LLM")
        logger.info(candidate_configs)
        logger.info("Ground truth for prompt configs")
        logger.info(candidate_fvals)
        logger.info('.'*50)

        # evaluate TPE_Ind
        if 'TPE_Ind' in to_evaluate:
            scores = fit_and_predict_with_TPE(hp_constraints, observed_configs, observed_fvals, candidate_configs, 0.5, multivariate=False, lower_is_better=lower_is_better)
            corr, regret = evaluate_score(scores, candidate_fvals, observed_fvals, lower_is_better, top_pct=0.5)
            logger.info(f'[TPE_Ind] Correlation: {corr:.4f}, Regret: {regret:.4f}')
            results['TPE_Ind']['corr'].append(corr)
            results['TPE_Ind']['regret'].append(regret)
            results['TPE_Ind']['y_prob'].append(scores.squeeze().tolist())
            results['TPE_Ind']['y_true'].append(candidate_fvals.to_numpy().squeeze().tolist())

        # evaluate TPE_Multi
        if 'TPE_Multi' in to_evaluate:
            scores = fit_and_predict_with_TPE(hp_constraints, observed_configs, observed_fvals, candidate_configs, 0.5, multivariate=True, lower_is_better=lower_is_better)
            corr, regret = evaluate_score(scores, candidate_fvals, observed_fvals, lower_is_better, top_pct=0.5)
            logger.info(f'[TPE_Multi] Correlation: {corr:.4f}, Regret: {regret:.4f}')
            results['TPE_Multi']['corr'].append(corr)
            results['TPE_Multi']['regret'].append(regret)
            results['TPE_Multi']['y_prob'].append(scores.squeeze().tolist())
            results['TPE_Multi']['y_true'].append(candidate_fvals.to_numpy().squeeze().tolist())

        # prepare task context for LLAMBO-family of evaluators
        task_context = {}
        task_context['model'] = model
        task_context['task'] = task_type
        task_context['tot_feats'] = dataset_loaded['train_x'].shape[1]
        task_context['cat_feats'] = 0
        task_context['num_feats'] = dataset_loaded['train_x'].shape[1]
        task_context['n_classes'] = len(np.unique(dataset_loaded['train_y']))
        task_context['metric'] = 'mean squared error' if task_metric == 'neg_mean_squared_error' else task_metric
        task_context['num_samples'] = dataset_loaded['train_x'].shape[0]
        task_context['hyperparameter_constraints'] = hp_constraints

        # evaluate LLAMBO - calibrated
        if "LLAMBO" in to_evaluate:
            LLM_SM = LLM_GEN_SM(task_context, n_gens=6, lower_is_better=lower_is_better, top_pct=0.2, n_templates=2, rate_limiter=rate_limiter, chat_engine=engine, logger=logger)
            _, pred_probs, cost, time_taken = LLM_SM.select_query_point(observed_configs, observed_fvals, candidate_configs, candidate_fvals, True)

            corr, regret = evaluate_score(pred_probs, candidate_fvals, observed_fvals, lower_is_better, top_pct=0.2)
            logger.info(f'[LLAMBO] Correlation: {corr:.4f}, Regret: {regret:.4f}, Cost: ${cost:.4f}, Time: {time_taken:.4f}s')
            results['LLAMBO']['corr'].append(corr)
            results['LLAMBO']['regret'].append(regret)
            results['LLAMBO']['y_prob'].append(pred_probs.squeeze().tolist())
            results['LLAMBO']['y_true'].append(candidate_fvals.to_numpy().squeeze().tolist())
            results['LLAMBO']['llm_query_cost'].append(cost)
            results['LLAMBO']['llm_query_time'].append(time_taken)
            tot_llm_cost += cost

        # evaluate LLAMBO - vanilla
        if "LLAMBO_VANILLA" in to_evaluate:
            LLM_SM = LLM_GEN_SM(task_context, n_gens=6, lower_is_better=lower_is_better, top_pct=0.2, n_templates=1, rate_limiter=rate_limiter, chat_engine=engine, logger=logger)
            _, pred_probs, cost, time_taken = LLM_SM.select_query_point(observed_configs, observed_fvals, candidate_configs, candidate_fvals, True)

            corr, regret = evaluate_score(pred_probs, candidate_fvals, observed_fvals, lower_is_better, top_pct=0.2)
            logger.info(f'[LLAMBO_VANILLA] Correlation: {corr:.4f}, Regret: {regret:.4f}, Cost: ${cost:.4f}, Time: {time_taken:.4f}s')
            results['LLAMBO_VANILLA']['corr'].append(corr)
            results['LLAMBO_VANILLA']['regret'].append(regret)
            results['LLAMBO_VANILLA']['y_prob'].append(pred_probs.squeeze().tolist())
            results['LLAMBO_VANILLA']['y_true'].append(candidate_fvals.to_numpy().squeeze().tolist())
            results['LLAMBO_VANILLA']['llm_query_cost'].append(cost)
            results['LLAMBO_VANILLA']['llm_query_time'].append(time_taken)
            tot_llm_cost += cost

        # save results
        with open(save_res_fpath, 'w') as f:
            json.dump(results, f, indent=4)

    logger.info('='*200)
    logger.info(f'[Evaluate Generative SM] {seed+1} evaluation runs complete! Total cost: ${tot_llm_cost:.4f}')

