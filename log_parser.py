import argparse
import re
import numpy as np

# Regexes that we'll get mileage out of
df_regex = re.compile(r".*?\d+.*?(\d\.\d+)")
llm_regex = re.compile(r".*?Response: \[(.*)\]")
llm_number_regex = re.compile(r".*?(\d.\d+)")

def logparse(logfile):
    print(f"Now parsing: {logfile}")
    log_sections = dict()
    with open(logfile,'r') as f:
        extracting = None
        for line in f.read().splitlines():
            # Determine if we can switch into processing a new section
            if extracting is None and "INFO - " in line:
                # Determine if we fall into the zone of the logs with the right data
                if "Observed_fvals (ICL)" in line:
                    extracting = 'icl'
                elif "Ground truth for prompt configs" in line:
                    extracting = 'ground_truth'
                elif "Query: " in line:
                    extracting = 'llm_responses'
                else:
                    continue
                # Prepare new section for data entry
                log_sections[extracting] = []
            # ICL and ground truth data are extracted the same way
            elif extracting == 'icl' or extracting == 'ground_truth':
                if "INFO - " in line and "score" in line:
                    continue
                number_extract = df_regex.findall(line)
                if number_extract is None or len(number_extract) == 0:
                    extracting = None
                else:
                    log_sections[extracting].append(float(number_extract[-1]))
            # Finally, parse out LLM responses
            elif extracting == 'llm_responses':
                if "Response:" in line:
                    llm_datapoints = [float(llm_number_regex.findall(_)[-1]) for _ in llm_regex.findall(line)[0].split(', ')]
                    log_sections[extracting].append(llm_datapoints)

    icl = np.asarray(log_sections['icl'])
    print(f"Identified {icl.shape} ICL values:")
    print(icl)
    ground_truth = np.asarray(log_sections['ground_truth'])
    print(f"Identified {ground_truth.shape} Ground Truth values for LLM to compare against")
    print(ground_truth)
    llm_responses = np.asarray(log_sections['llm_responses'])
    print(f"Identified {llm_responses.shape} LLM responses")
    print(llm_responses)
    # LLAMBO evaluates itself this way
    average_difference = np.abs(llm_responses.mean(axis=1) - ground_truth).mean()
    print(f"Average difference of (MEAN LLM response per query vs query's ground truth) per Ground Truth")
    print(average_difference)
    # I want to know if the LLM made unique values
    llm_flat = llm_responses.ravel().tolist()
    unique_llm_generations = set(llm_flat).difference(set(icl))
    print("Set of unique values produced by the LLM and how many times they appeared")
    print(dict((k, llm_flat.count(k)) for k in unique_llm_generations))
    plagiarism_dictionary = dict((k,llm_flat.count(k)) for k in sorted(set(icl)) if llm_flat.count(k) > 0)
    # I want to know if the LLM stole values from ICL and how often
    print("Set of copied values from ICL that the LLM used and how many times it used them")
    print(plagiarism_dictionary)

prs = argparse.ArgumentParser()
prs.add_argument("logs", nargs="+", default=None, help="Logfiles to parse")
args = prs.parse_args()

for logfile in args.logs:
    logparse(logfile)

