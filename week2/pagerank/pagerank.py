import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    #if len(sys.argv) != 2:
        #sys.exit("Usage: python pagerank.py corpus")
    #corpus = crawl(sys.argv[1])
    corpus = crawl("corpus0")
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    dist = {}
    for cur_page in corpus[page]:
        dist[cur_page] = damping_factor/len(corpus[page])
    for any_page in corpus:
        if any_page not in dist:
            dist[any_page] = 0
        dist[any_page] += (1-damping_factor)/len(corpus)
    return dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sample_dict = {}
    for key in corpus:
        sample_dict[key] = 0
    cur_sample_page = None
    for x in range(n):
        if x == 0:
            cur_sample_page = random.choice(list(sample_dict.keys()))
        else:
            dist = transition_model(corpus, cur_sample_page, DAMPING)
            dist_weights = [val for val in dist.values()]
            cur_sample_page = random.choices(list(dist.keys()), dist_weights)[0]
        sample_dict[cur_sample_page] += (1/n)
    return sample_dict
    

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pr_dict = {}
    for key in corpus:
        pr_dict[key] = 1/len(corpus)
    
    max_change = float('-inf')
    while(max_change > 0.001):
        cur_max = float('-inf')
        for page in corpus:
            new_pr = PR(damping_factor, corpus, page)
            cur_max = max(cur_max, abs(pr_dict[page]-new_pr))
            pr_dict[page] = new_pr
        max_change = cur_max
    
    return pr_dict

def PR(damping_factor, corpus, cur_page):
    first_cond = (1-damping_factor)/len(corpus)
    second_cond = 0
    for page in corpus[cur_page]:
        if len(corpus[cur_page] == 0):
            second_cond += PR(damping_factor, corpus, page) / len(corpus)
        else:
            second_cond += PR(damping_factor, corpus, page) / len(corpus[page])
    second_cond *= damping_factor
    return (first_cond + second_cond)


if __name__ == "__main__":
    main()
