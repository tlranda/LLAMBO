import argparse
import pprint
from langchain_community.llms import ollama
from transformers import GPT2TokenizerFast
import matplotlib
font = {'size': 8}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("-files", nargs="+", required=True, default=None, help="Text files to parse (each line is formatted as '#|#')")
    prs.add_argument("-title", default=None, help="Plot title (defaults to filenames)")
    prs.add_argument("-save", default=None, help="File to save image to (default: do not save, just display)")
    prs.add_argument("--special-branching-colors", action="store_true", help="Attempt to apply consistent coloring scheme to initial color stems (default: %(default)s)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    return args

def get_tokenizer():
    #llm = ollama.Ollama(model='llama3')
    #return llm.get_num_tokens
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    return tokenizer

def adjust_lightness_hsv(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.rgb_to_hsv(mcolors.to_rgb(c))
    c[2] = max(0, min(1, amount*c[2]))
    return mcolors.to_hex(c)

def adjust_lightness(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = np.asarray(mcolors.to_rgb(c))
    c = np.clip(c * amount, 0, 1)
    return mcolors.to_hex(c)
    
def apply_gradient(color_string, n_total, tightness=0.5):
    #print(f"Applying gradient to color {color_string} x{n_total}")
    if n_total < 3:
        #print("Too small, return original color")
        return [color_string] * n_total
    mid_index = n_total//2
    gradient_colors = []
    for i in range(n_total):
        factor = 1
        if i < mid_index:
            # Darker
            factor -= tightness * ((mid_index - i) / (mid_index + 1))
        elif i > mid_index:
            factor += tightness * ((i - mid_index) / (n_total - mid_index - 1))
        gradient_colors.append(adjust_lightness(color_string, factor))
    #print(f"New colors: {gradient_colors}")
    return gradient_colors

def main(args=None):
    args = parse(args)
    tokenizer = get_tokenizer()

    for file in args.files:
        with open(file,'r') as f:
            to_parse = f.read().splitlines()
        substrings_with_n_tokens = dict()

    for line in to_parse:
        line, n_times = line.rsplit('|',1)
        n_times = int(n_times)
        #print(line, "appears", n_times, "times")
        paraphrase = f"## {line} ##"
        tokens = [tokenizer.decode(_) for _ in tokenizer.encode_plus(paraphrase)['input_ids']]
        if tokens[3] != "00":
            continue
        psubstr = ""
        substr = ""
        for idx, tok in enumerate(tokens):
            substr += tok
            if idx+1 in substrings_with_n_tokens:
                if substr in substrings_with_n_tokens[idx+1]:
                    substrings_with_n_tokens[idx+1][substr][0] += n_times
                else:
                    substrings_with_n_tokens[idx+1][substr] = [n_times, psubstr]
            else:
                substrings_with_n_tokens[idx+1] = {substr: [n_times, psubstr]}
            psubstr = substr
        """
        last_n_tokens = 0
        last_ext = 0
        pprev = None
        prev = ""
        for ext in range(1,len(paraphrase)+1):
            subset = paraphrase[:ext]
            n_tokens = tokenizer(subset)
            if n_tokens > last_n_tokens:
                #print("\t", n_tokens-1, f"'{prev}'")
                if n_tokens-1 in substrings_with_n_tokens:
                    if prev in substrings_with_n_tokens[n_tokens-1]:
                        substrings_with_n_tokens[n_tokens-1][prev][0] += n_times
                    else:
                        substrings_with_n_tokens[n_tokens-1][prev] = [n_times, pprev]
                else:
                    substrings_with_n_tokens[n_tokens-1] = {prev: [n_times, pprev]}
                last_n_tokens = n_tokens
                last_ext = ext
                pprev = prev
            prev = subset
        if last_ext < len(paraphrase):
            #print("\t", n_tokens, f"'{subset}'")
            if n_tokens in substrings_with_n_tokens:
                if subset in substrings_with_n_tokens[n_tokens]:
                    substrings_with_n_tokens[n_tokens][subset][0] += n_times
                else:
                    substrings_with_n_tokens[n_tokens][subset] = [n_times, pprev]
            else:
                substrings_with_n_tokens[n_tokens] = {subset: [n_times, pprev]}
        """
    pprint.pprint(substrings_with_n_tokens)

    # Largest width that needs processing determines center height
    center_height = max(map(len, substrings_with_n_tokens.values()))/2
    fig, ax = plt.subplots()
    prev_ys = {'': center_height}
    saved_colors = None
    origin_colors = None
    for token_length in sorted(substrings_with_n_tokens.keys()):
        if token_length == 0:
            continue
        # Fetch current substrings and frequencies, and sort them based on hi-to-lo frequency
        substrs_freq = substrings_with_n_tokens[token_length]
        freqs = np.asarray([_[0] for _ in substrs_freq.values()])
        # Use just the most recently added token
        prev_substrs = np.asarray([v[1] for v in substrs_freq.values()])
        untrimmed_substrs = np.asarray(list(substrs_freq.keys()))
        trimmed_substrs = np.asarray([f"{k[len(v[1]):]} ({v[0]})" for (k,v) in substrs_freq.items()])
        raw_trimmed_substrs = np.asarray([k[len(v[1]):] for (k,v) in substrs_freq.items()])

        # Determine sorting order:
        # - Strings that share a common substring should be sorted together based on that substring
        # - Within the same common substring, higher frequency should be sorted higher
        sort_order = []
        # Indices we need to bucket-ize
        unbucketed = range(len(prev_substrs))
        # How we previously sorted for same traversal order
        prev_sort = np.argsort(list(prev_ys.values()))
        if saved_colors is not None:
            try:
                colors = [saved_colors[ut] for ut in prev_substrs]
            except:
                colors = ['k'] * len(prev_substrs)
        elif len(trimmed_substrs) == 1:
            colors = ['k'] * len(prev_substrs)
        else:
            if args.special_branching_colors:
                colorbranches = {'001': 'orange', '002': 'red', '000': 'green', '0010': 'olive'}
                mcolorbranches = dict((k,mcolors.TABLEAU_COLORS[f'tab:{v}']) for (k,v) in colorbranches.items())
                colors = [mcolorbranches[k] if k in mcolorbranches else None for k in raw_trimmed_substrs]
            else:
                colors = [None] * len(prev_substrs)
        colors = np.asarray(colors)
        for key in np.asarray(list(prev_ys.keys()))[prev_sort]:
            #print(f"Sorting key {key}")
            new_bucket = []
            for idx in unbucketed:
                if prev_substrs[idx] == key:
                    new_bucket.append(idx)
            #unbucketed = list(filter(lambda x: x not in new_bucket, unbucketed))
            unbucketed = [_ for _ in unbucketed if _ not in new_bucket]
            # Go ahead and sort the current bucket based on frequency
            new_bucket_freqs = freqs[new_bucket]
            within_bucket_sort = np.argsort(new_bucket_freqs)
            new_bucket = np.asarray(new_bucket)[within_bucket_sort]
            if saved_colors is not None and len(new_bucket) > 0 and colors[new_bucket[0]] is not None and colors[new_bucket[0]] != 'k':
                colors[new_bucket] = apply_gradient(colors[new_bucket[0]], len(new_bucket))
            sort_order.extend(new_bucket.tolist())
        # Apply sorting
        freqs = freqs[sort_order]
        prev_substrs = prev_substrs[sort_order]
        untrimmed_substrs = untrimmed_substrs[sort_order]
        trimmed_substrs = trimmed_substrs[sort_order]
        colors = colors[sort_order]
        saved_colors = None
        # Determine how many items are at this portion of the plot
        n_items = len(trimmed_substrs)
        base_height = center_height / n_items
        xs = [token_length] * n_items
        ys = [center_height+(base_height*index) for index in range(-(n_items//2),(n_items//2)+(n_items%2==1))]
        if origin_colors is None and colors[0] is None:
            origin_colors = np.asarray([mcolors.to_rgb(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(trimmed_substrs)]])
        # Add the backwards-connecting line segment and text
        for xx, yy, prev_sub, ut, tt, color in zip(xs, ys, prev_substrs, untrimmed_substrs, trimmed_substrs, colors):
            line_segment = ax.plot([xx-1,xx],[prev_ys[prev_sub], yy], color=color, linestyle='dotted')
            if color != 'k':
                if saved_colors is None:
                    saved_colors = {ut: line_segment[0].get_color()}
                else:
                    saved_colors[ut] = line_segment[0].get_color()
            # Find closest color to you in origin colors
            if origin_colors is None:
                tcolor = 'k'
            else:
                tcolor = mcolors.to_hex(origin_colors[np.argmin(np.abs(origin_colors - mcolors.to_rgb(line_segment[0].get_color())).sum(axis=1)),:])
            text = ax.text(xx,yy,tt,ha='center',va='center',zorder=10, color=tcolor)
            text.set_bbox({'facecolor':'lightgray','alpha':0.8,'edgecolor':'none'})
        prev_ys = dict((k,v) for (k,v) in zip(untrimmed_substrs,ys))
    if args.title is None:
        ax.set_title(",".join(args.files))
    else:
        ax.set_title(args.title)
    orientations = ['bottom','top','left','right']
    orientations.extend([f'label{k}' for k in orientations])
    plt.tick_params(axis='both',which='both',**dict((k,False) for k in orientations))
    plt.tight_layout()
    if not args.save:
        plt.show()
    else:
        fig.savefig(args.save, dpi=300)

if __name__ == "__main__":
    main()

