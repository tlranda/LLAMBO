import os
import time
import openai
import asyncio
import re
import numpy as np
import pandas as pd
from aiohttp import ClientSession
from llambo.rate_limiter import RateLimiter
from llambo.generative_sm_utils import gen_prompt_tempates

openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_version = os.environ["OPENAI_API_VERSION"]
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]


class LLM_GEN_SM:
    ollama_engines = ['llama3','llama3.1','phi3']
    def __init__(self, task_context, n_gens, lower_is_better, top_pct,
                 n_templates=1, rate_limiter=None, 
                 verbose=False, chat_engine=None, logger=None):
        '''Initialize the forward LLM surrogate model. This is modelling p(y|x) as in GP/SMAC etc.'''
        self.task_context = task_context
        self.n_gens = n_gens
        self.lower_is_better = lower_is_better
        self.top_pct = top_pct
        self.n_templates = n_templates
        if rate_limiter is None:
            self.rate_limiter = RateLimiter(max_tokens=240000, time_frame=60, max_requests=2900)
        else:
            self.rate_limiter = rate_limiter
        if logger is None:
            import logging
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.recalibrator = None
        self.chat_engine = chat_engine
        self.verbose = verbose

    async def _async_generate(self, few_shot_template, query_example, query_idx):
        '''Generate a response from the LLM async.'''
        prompt = few_shot_template.format(Q=query_example['Q'])

        MAX_RETRIES = 3

        async with ClientSession(trust_env=True) as session:
            openai.aiosession.set(session)

            resp = None
            n_preds = int(self.n_gens/self.n_templates)
            n_generations = max(n_preds,3)
            for retry in range(MAX_RETRIES):
                try:
                    self.rate_limiter.add_request(request_text=prompt, current_time=time.time())
                    openai_kwargs = {'prompt': prompt,
                                     'temperature': 0.7,
                                     'top_p': 0.95,
                                     'max_tokens': 8,
                                     'request_timeout': None, # Formerly 10
                                     }
                    if self.chat_engine in self.ollama_engines:
                        # Ollama needs this to be the MODEL argument
                        openai_kwargs.update({'model': self.chat_engine})
                    else:
                        # OpenAI API needs this to be the ENGINE argument
                        openai_kwargs.update({'engine': self.chat_engine})
                        # OpenAI API can generate mutliple responses on this call
                        openai_kwargs.update({'n': n_generations})
                        # OpenAI API supports a logprobs argument that LLAMBO used
                        # Include the log probabilities on the logprobs most likely output tokens, as well the chosen tokens. For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response. The maximum value for logprobs is 5.
                        openai_kwargs.update({'logprobs': 5})
                    resp = await openai.Completion.acreate(**openai_kwargs)
                    self.rate_limiter.add_request(request_token_count=resp['usage']['total_tokens'], current_time=time.time())
                    # Using Ollama, call multiple times for n_generations
                    other_resps = []
                    if self.chat_engine in self.ollama_engines:
                        for i in range(n_generations-1):
                            self.rate_limiter.add_request(request_text=prompt, current_time=time.time())
                            # Ensure seeds change so you don't always get the exact same response
                            if 'seed' in openai_kwargs:
                                openai_kwargs['seed'] += 1
                            bonus_resp = await openai.Completion.acreate(**openai_kwargs)
                            self.rate_limiter.add_request(request_token_count=bonus_resp['usage']['total_tokens'], current_time=time.time())
                            other_resps.append(bonus_resp)
                    # Merge all responses
                    new_index = len(resp.choices)
                    for bonus in other_resps:
                        choices_index = list(bonus.keys()).index('choices')
                        usage_index = list(bonus.keys()).index('usage')
                        choices = list(bonus.values())[choices_index]
                        usage = list(bonus.values())[usage_index]
                        for use_key, use_value in usage.items():
                            setattr(resp.usage,use_key,getattr(resp.usage,use_key)+use_value)
                        for choice in choices:
                            choice.index = new_index
                            new_index += 1
                        resp.choices.extend(choices)
                    break
                except Exception as e:
                    self.logger.info(f"LLM generation attempt {retry+1}/{MAX_RETRIES} failed:"+str(e))
                    print(f'[SM] RETRYING LLM REQUEST {retry+1}/{MAX_RETRIES}...')
                    print(resp)
                    print(e)
                    if retry == MAX_RETRIES-1:
                        await openai.aiosession.get().close()
                        raise e
                    pass

        await openai.aiosession.get().close()

        if resp is None:
            return None

        tot_tokens = resp['usage']['total_tokens']
        tot_cost = 0.0015*(resp['usage']['prompt_tokens']/1000) + 0.002*(resp['usage']['completion_tokens']/1000)

        return query_idx, resp, tot_cost, tot_tokens

    async def _generate_concurrently(self, few_shot_templates, query_examples):
        '''Perform concurrent generation of responses from the LLM async.'''
        coroutines = []
        for template in few_shot_templates:
            for query_idx, query_example in enumerate(query_examples):
                coroutines.append(self._async_generate(template, query_example, query_idx))

        results = [[] for _ in range(len(query_examples))]      # nested list
        # My machine cannot locally generate all of these at once, rewrite to be one-at-a-time please
        for c in coroutines:
            task = asyncio.create_task(c)
            llm_response = await asyncio.gather(task)
            for response in llm_response:
                if response is not None:
                    query_idx, resp, tot_cost, tot_tokens = response
                    results[query_idx].append([resp, tot_cost, tot_tokens])
        # Old version that would generate all templates in parallel
        """
        tasks = [asyncio.create_task(c) for c in coroutines]
        llm_response = await asyncio.gather(*tasks)

        for response in llm_response:
            if response is not None:
                query_idx, resp, tot_cost, tot_tokens = response
                results[query_idx].append([resp, tot_cost, tot_tokens])
        """
        return results  # format [(resp, tot_cost, tot_tokens), None, (resp, tot_cost, tot_tokens)]

    def process_response(self, all_raw_response):
        all_pred_probs = [] # p(s<\tau | h)
        for raw_response in all_raw_response:
            tokens = raw_response['tokens']
            logprobs = raw_response['top_logprobs']
            pred_index = min((tokens.index(val) for val in ["0", "1"] if val in tokens), default=None)
            if pred_index is None:
                all_pred_probs.append(np.nan)
            else:
                try:
                    prob_1 = logprobs[pred_index]["1"]
                    prob_0 = logprobs[pred_index]["0"]
                    prob_1 = np.exp(prob_1)/(np.exp(prob_1) + np.exp(prob_0))
                    all_pred_probs.append(prob_1)
                except:
                    all_pred_probs.append(np.nan)

        return all_pred_probs

    
    async def _predict(self, all_prompt_templates, query_examples, query_ground_truths):
        start = time.time()
        all_preds = []
        tot_tokens = 0
        tot_cost = 0

        bool_pred_returned = []

        # make predictions in chunks of 5, for each chunk make concurent calls
        n_concurrent = 1 #5
        #for i in range(0, len(query_examples), n_concurrent):
        i = 0
        while i < len(query_examples):
            query_chunk = query_examples[i:i+n_concurrent]
            query_truths = query_ground_truths[i:i+n_concurrent]
            self.logger.info(f"Query: {query_chunk}")
            self.logger.info(f"Ground Truth: {query_truths}")
            print(f"Query: {query_chunk}")
            print(f"Ground Truth: {query_truths}")
            chunk_results = await self._generate_concurrently(all_prompt_templates, query_chunk)
            bool_pred_returned.extend([1 if x is not None else 0 for x in chunk_results])                # track effective number of predictions returned

            for _, sample_response in enumerate(chunk_results):
                if not sample_response:     # if sample prediction is an empty list :(
                    sample_preds = [np.nan] * self.n_gens
                else:
                    sample_preds = []
                    try:
                        all_raw_response = [x['logprobs'] for template_response in sample_response for x in template_response[0]['choices'] ]        # fuarr this is some high level programming
                        sample_preds = self.process_response(all_raw_response)
                    except KeyError:
                        # Ollama doesn't support logprobs, so it won't be present
                        all_raw_response = [x['text'] for template_response in sample_response for x in template_response[0]['choices'] ]        # fuarr this is some high level programming
                        for gen_text in all_raw_response:
                            gen_pred = re.findall(r"## ?(-?\d+) ?##", gen_text)
                            if len(gen_pred) == 1:
                                sample_preds.append(float(gen_pred[0]))
                            else:
                                sample_preds.append(np.nan)
                        while len(sample_preds) < self.n_gens:
                            sample_preds.append(np.nan)
                    self.logger.info(f"Response: {all_raw_response}")
                    print(f"Response: {all_raw_response}")
                    tot_cost += sum([x[1] for x in sample_response])
                    tot_tokens += sum([x[2] for x in sample_response])
                all_preds.append(sample_preds)
            # This deletion may need to be re-examined if I turn on concurrent responses again
            if np.isnan(all_preds).all():
                del bool_pred_returned[-n_concurrent:]
            else:
                i += n_concurrent
        end = time.time()
        time_taken = end - start

        pred_probs = np.array(all_preds).astype(float)
        # Sometimes one or more row may have nothing except for nan values
        has_atleast_one_value = np.argwhere(~np.isnan(pred_probs).all(axis=1)).ravel()
        has_all_nans = np.argwhere(np.isnan(pred_probs).all(axis=1)).ravel()
        success_rate = sum(bool_pred_returned)/len(bool_pred_returned)

        ok_preds = np.take(pred_probs, has_atleast_one_value, axis=0)
        y_mean = np.nanmean(ok_preds, axis=1)
        y_std = np.nanstd(ok_preds, axis=1)

        # Capture failed calls - impute None with average predictions
        y_mean[np.isnan(y_mean)] = np.nanmean(y_mean)
        y_std[np.isnan(y_std)]   = np.nanmean(y_std)
        y_std[y_std<1e-5] = 1e-5 # replace small values to avoid division by zero

        return y_mean, success_rate, tot_cost, tot_tokens, time_taken
    
    async def _evaluate_candidate_points(self, observed_configs, observed_fvals, candidate_configs, candidate_fvals):
        '''Evaluate candidate points using the LLM model.'''
        all_run_cost = 0
        all_run_time = 0

        all_prompt_templates, query_examples = gen_prompt_tempates(self.task_context, observed_configs, observed_fvals, candidate_configs, 
                                                                   self.lower_is_better, self.top_pct, n_prompts=self.n_templates)
        
        self.logger.info('*'*100)
        self.logger.info(f'Number of all_prompt_templates: {len(all_prompt_templates)}')
        self.logger.info(f'Number of query_examples: {len(query_examples)}')
        self.logger.info(all_prompt_templates[0].format(Q=query_examples[0]['Q']))
        print('*'*100)
        print(f'Number of all_prompt_templates: {len(all_prompt_templates)}')
        print(f'Number of query_examples: {len(query_examples)}')
        print(all_prompt_templates[0].format(Q=query_examples[0]['Q']))


        response = await self._predict(all_prompt_templates, query_examples, candidate_fvals)

        pred_probs, success_rate, tot_cost, tot_tokens, time_taken = response

        all_run_cost += tot_cost
        all_run_time += time_taken

        return pred_probs, all_run_cost, all_run_time


    def _warp_candidate_points(self, configurations):
        '''Warp candidate points to log scale if necessary.'''
        warped_configs = configurations.copy().to_dict(orient='records')
        hyperparameter_constraints = self.task_context['hyperparameter_constraints']
        for config in warped_configs:
            for hyperparameter, constraint in hyperparameter_constraints.items():
                if constraint[1] == 'log':
                    config[hyperparameter] = np.log10(config[hyperparameter])

        warped_configs = pd.DataFrame(warped_configs)
        return warped_configs
    

    def _unwarp_candidate_points(self, configurations):
        '''Unwarp candidate points from log scale if necessary.'''
        unwarped_configs = configurations.copy().to_dict(orient='records')
        hyperparameter_constraints = self.task_context['hyperparameter_constraints']
        for config in unwarped_configs:
            for hyperparameter, constraint in hyperparameter_constraints.items():
                if constraint[1] == 'log':
                    config[hyperparameter] = 10**config[hyperparameter]

        unwarped_configs = pd.DataFrame(unwarped_configs)
        return unwarped_configs
    

    def select_query_point(self, observed_configs, observed_fvals, candidate_configs, candidate_fvals, return_raw_preds=False):
        '''Select the next query point using expected improvement.'''
        # warp candidate points
        observed_configs = self._warp_candidate_points(observed_configs)
        candidate_configs = self._warp_candidate_points(candidate_configs)

        pred_probs, cost, time_taken = asyncio.run(self._evaluate_candidate_points(observed_configs, observed_fvals, candidate_configs, candidate_fvals))

        best_point_index = np.argmax(pred_probs)

        # unwarp candidate points
        candidate_configs = self._unwarp_candidate_points(candidate_configs)

        best_point = candidate_configs.iloc[[best_point_index], :]  # return selected point as dataframe not series

        if return_raw_preds:
            return best_point, pred_probs, cost, time_taken
        else:
            return best_point, cost, time_taken

