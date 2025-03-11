import sys
import math
import random
import itertools
from collections import namedtuple, defaultdict, Counter

import tqdm

Rule = namedtuple("Rule", ['lhs', 'rhs'])

ROOT = 'ROOT'
NONE = '-NONE-'
TERMINAL_MARKER = '_'

class PCFG:
    """ PCFG in Chomsky Normal Form. """
    def __init__(self, nonterminal_rules, terminal_rules, root):
        self.nt_rules = nonterminal_rules
        self.t_rules = terminal_rules
        self.root = root

        # Build mappings from rhs to rules with probabilities
        self.nt_rules_inv = {}
        self.t_rules_inv = {}
        for rule, p in self.t_rules.items():
            self.t_rules_inv.setdefault(rule.rhs[0], []).append((rule.lhs, p))
        for rule, p in self.nt_rules.items():
            self.nt_rules_inv.setdefault(rule.rhs, []).append((rule.lhs, p))
        
    def score(self, xs):
        T = len(xs)
        chart = [[{} for _ in range(T)] for _ in range(T)]
        for i, word in enumerate(xs):
            for nt, p in self.t_rules_inv[word]:
                chart[i][i][nt] = p
        for span in range(2, T+1):
            for i in range(T - span + 1):
                j = i + span - 1
                cell = Counter()
                for k in range(i, j):
                    left_cell = chart[i][k]
                    right_cell = chart[k+1][j]
                    for B, bscore in left_cell.items():
                        for C, cscore in right_cell.items():
                            for nt, p in self.nt_rules_inv.get((B, C), []):
                                cell[nt] += bscore * cscore * p
                chart[i][j] = cell
        return math.log(chart[0][T-1][self.root])

def gensym(_state=itertools.count()):
    return 'X' + str(next(_state))

def preterminal_for(x):
    return 'P' + x

def nonterminal_for_sequence(xs, rules, _suffix_dict={}):
    if xs in _suffix_dict:
        return _suffix_dict[xs]
    elif len(xs) == 2:
        new_nt = gensym()
        rule = Rule(new_nt, (xs[0], xs[1]))
        rules[rule] = 1.0
        _suffix_dict[xs] = new_nt
        return new_nt
    else:
        new_nt = gensym()
        first, *rest = xs
        next_nt = nonterminal_for_sequence(tuple(rest), rules)
        rule = Rule(new_nt, (first, next_nt))
        rules[rule] = 1.0
        _suffix_dict[xs] = new_nt
        return new_nt

def is_terminal(symbol):
    return symbol.startswith(TERMINAL_MARKER) or symbol == NONE

def convert_to_cnf(rules):
    # remove unary productions
    nonterminals = {rule.lhs for rule in rules.keys()}
    unit_paths = defaultdict(Counter)
    for A in nonterminals:
        unit_paths[A][A] = 1.0
        queue = [A]
        while queue:
            x = queue.pop()
            for rule, p in rules.items():
                if rule.lhs == x and len(rule.rhs) == 1:
                    y, = rule.rhs
                    new_prob = unit_paths[A][x] * p
                    if y not in unit_paths[A]:
                        unit_paths[A][y] = new_prob
                        queue.append(y)
    deunarized_rules = Counter()
    for A in nonterminals:
        for B in unit_paths[A]:
            for rule, p in rules.items():
                if rule.lhs == B and (len(rule.rhs) > 1 or is_terminal(rule.rhs[0])):
                    new_prob = unit_paths[A][B] * p
                    new_rule = Rule(A, rule.rhs)
                    deunarized_rules[new_rule] += new_prob
        
    # add preterminals
    pt_rules = {}
    for (lhs, rhs), p in deunarized_rules.items():
        new_rhs = []
        for symbol in rhs:
            assert len(rhs) > 1 or is_terminal(rhs[0])
            if is_terminal(symbol) and len(rhs) > 1:
                preterminal = preterminal_for(symbol)
                new_preterminal_rule = Rule(preterminal, (symbol,))
                pt_rules[new_preterminal_rule] = 1.0
                new_rhs.append(preterminal)
            else:
                new_rhs.append(symbol)
        pt_rules[Rule(lhs, tuple(new_rhs))] = p

    # binarize by introducing new nonterminals
    nt_rules = {}
    t_rules = {}
    for rule, p in pt_rules.items():
        if len(rule.rhs) == 1: # terminal rule
            t_rules[rule] = p
        elif len(rule.rhs) == 2: # binary rule
            nt_rules[rule] = p
        else: # ternary+ rule
            first, *rest = rule.rhs
            new_nt = nonterminal_for_sequence(tuple(rest), nt_rules)
            rule = Rule(rule.lhs, (first, new_nt))
            nt_rules[rule] = p

    return nt_rules, t_rules

def test():
    for i in range(100):
        p_continue = random.random() * .4
        rules = {
            Rule('S', ('_a', '_b')) : 1 - p_continue,
            Rule('S', ('_a', 'S', '_b')) : p_continue,
        }
        nt_rules, t_rules = convert_to_cnf(rules)
        pcfg = PCFG(nt_rules, t_rules, 'S')
        assert pcfg.score(['_a', '_b']) == math.log(1 - p_continue)
        assert pcfg.score(['_a', '_a', '_b', '_b']) == math.log(p_continue * (1 - p_continue))
        assert pcfg.score(['_a', '_a', '_a', '_b', '_b', '_b']) == math.log(p_continue **2 * (1 - p_continue))
        
def read_grammar(grammar_filename):
    rules = {}
    with open(grammar_filename) as infile:
        for line in infile:
            if line.startswith('#'):
                continue
            else:
                logprob, lhs, *rhs = line.strip().split()
                if rhs:
                    rule = Rule(lhs, tuple(rhs))
                    rules[rule] = math.exp(float(logprob))
    nt_rules, t_rules = convert_to_cnf(rules)
    return PCFG(nt_rules, t_rules, ROOT)

def main(grammar_filename, text_filename):
    print("Processing grammar...", file=sys.stderr)
    grammar = read_grammar(grammar_filename)
    print("Built CNF grammar with %d nonterminal rules." % len(grammar.nt_rules), file=sys.stderr)
    print("Calculating inside probabilities...", file=sys.stderr)    
    with open(text_filename) as infile:
        lines = infile.readlines()
    for line in tqdm.tqdm(lines):
        terminals = [
            "".join([TERMINAL_MARKER, terminal]) if terminal != NONE else terminal
            for terminal in line.strip().split()
        ]
        score = grammar.score(terminals)
        print(line.strip(), "\t", score, sep="")

if __name__ == '__main__':
    main(*sys.argv[1:])
