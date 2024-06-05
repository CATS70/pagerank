import copy
import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
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


def get_numlinks_by_page(corpus):
    numlinkdict = {}
    for k, v in corpus.items():
        if len(v) != 0:
            numlinkdict[k] = len(v)
        else:
            numlinkdict[k] = len(corpus)
    return numlinkdict


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # raise NotImplementedError
    # set variables
    corpuspages = list(corpus.keys())
    nbcorpuspages = len(corpuspages)
    # linkedpages =list of pages linked to the page
    linkedpages = corpus[page]

    # nbpage is the number of linked pages
    nbpage = len(linkedpages)
    # addp is the additional probability
    addp = (1-damping_factor)/nbcorpuspages

    plinkedpages = dict()

    if len(linkedpages) == 0:
        p = 1/nbcorpuspages
        for lpage in corpuspages:
            plinkedpages[lpage] = p
        return plinkedpages

    # test if page is in the linked pages
    for pg in corpuspages:
        if {pg}.issubset(linkedpages):
            p = 1 / nbpage
            plinkedpages[pg] = p + addp
        else:
            plinkedpages[pg] = addp
    return plinkedpages


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # result will contain sampling results
    result = dict.fromkeys(list(corpus.keys()), 0)

    # choose the 1st page and add in the result dict
    listofpage = list(result)
    page = [random.choice(listofpage)]

    result[page[0]] += 1

    for i in range(n-1):
        # get the transition model
        tm = transition_model(corpus, page[0], damping_factor)
        # extract the probabilities from tm
        pvalues = list(tm.values())
        # extract page's name form tm
        pgnames = list(tm.keys())
        # Choose randomly the page according to the page's weight

        page = random.choices(pgnames, weights=pvalues, k=1)

        result[page[0]] += 1

    # Result calculation
    tempresult = copy.deepcopy(result)
    for pg, pgv in tempresult.items():
        result[pg] = pgv/n
    return result


def PR(page, lcorpus, numlinks, cresults, d):
    nbpage = len(lcorpus)
    pr = (1-d)/nbpage
    for lpage in lcorpus[page]:
        pr += d*cresults[lpage]['cur']/numlinks[lpage]

    return pr


def corpus_link(corpus):
    corpusl = {}
    # search if a page has no link and, if so,
    # add all pages as links including itself
    for pg, pgl in corpus.items():
        if len(pgl) == 0:
            corpus[pg] = corpus.keys()

    # For each page link create or update an entry in a dict
    # with pages pointing to that page as value
    for pg, pgl in corpus.items():
        for page in pgl:
            if not page in corpusl:
                corpusl[page] = {pg}
            else:
                myset = corpusl[page]
                myset.update({pg})
                corpusl[page] = myset
    return corpusl


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # 1 - prepare date : pages, numlink, result dictionaries

    numlinksbypage = get_numlinks_by_page(corpus)
    linkscorpus = corpus_link(corpus)

    pagenumber = len(linkscorpus)
    prresults = {}
    prfinalresults = {}
    # to stop while loop when all pr eval will be stable
    flags = {}
    # var init
    for k, v in linkscorpus.items():
        prresults[k] = {'prev': 0, 'cur': 1/pagenumber}
        prfinalresults[k] = 0
        flags[k] = False
    # 2 - run calculation
    while True:
        for k, v in linkscorpus.items():
            curpr = PR(k, linkscorpus, numlinksbypage, prresults, damping_factor)
            prevpr = prresults[k]['prev']
            prresults[k]['prev'] = prresults[k]['cur']
            prresults[k]['cur'] = curpr
            if abs(curpr-prevpr) <= 0.001:
                flags[k] = True
        if sum(flags.values()) == pagenumber:
            break

    for k, v in prresults.items():
        prfinalresults[k] = prresults[k]['cur']

    return prfinalresults


if __name__ == "__main__":
    main()