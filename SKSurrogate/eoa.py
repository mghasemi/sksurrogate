"""
Evolutionary Optimization Algorithm
===========================================
"""


class EOA(object):
    """
    This is a base class acting as an umbrella to process an
    evolutionary optimization algorithm.

    :param population: The whole possible population as a list
    :param fitness: The fitness evaluation. Accepts an OrderedDict of individuals with their corresponding fitness and
        updates their fitness
    :param init_pop: default=`UniformRand`; The python class that initiates the initial population
    :param recomb: default=`UniformCrossover`; The python class that defines how to combine parents to produce children
    :param mutation: default=`Mutation`; The python class that performs mutation on offspring population
    :param termination: default=`MaxGenTermination`; The python class that determines the termination criterion
    :param elitism: default=`Elites`; The python class that decides how to handel elitism
    :param num_parents: The size of initial parents population
    :param parents_porp: default=0.1; The size of initial parents population given as a portion of whole population
        (only used if `num_parents` is not given)
    :param elits_porp: default=0.2; The porportion of offspring to be replaced by elite parents
    :param mutation_prob: The probability that a component will be mutated (default: 0.05)
    :param kwargs:
    """

    def __init__(self, population, fitness, **kwargs):

        from collections import OrderedDict

        self.population = population
        self.init_pop = kwargs.pop("init_pop", UniformRand)(**kwargs)
        self.fitness = fitness
        self.recomb = kwargs.pop("recomb", UniformCrossover)(**kwargs)
        self.mutation = kwargs.pop("mutation", Mutation)(**kwargs)
        self.termination = kwargs.pop("termination", MaxGenTermination)(**kwargs)
        self.elitism = kwargs.pop("elitism", Elites)(**kwargs)
        self.population_size = len(self.population)
        self.parents_porp = kwargs.pop("parents_porp", 0.1)
        self.num_parents = kwargs.pop(
            "num_parents", 2 * int(self.population_size * self.parents_porp / 2.0)
        )
        self.elits_porp = kwargs.pop("elits_porp", 0.2)
        self.num_elites = int(self.elits_porp * self.num_parents)
        self.mutation_prob = kwargs.pop("mutation_prob", 0.05)
        self.max_generations = kwargs.pop("max_generation", 50)
        self.generation_num = 0
        self.genes = kwargs.pop("genes", [])
        self.init_genes = kwargs.pop("init_genes", [])
        self.term_genes = kwargs.pop("term_genes", [])
        self.task_name = kwargs.pop("task_name", "EOA")
        self.check_point = kwargs.pop("check_point", "./")
        if not self.genes:
            self.find_genes()
        self.evals = OrderedDict([(_, None) for _ in self.population])
        self.parents = OrderedDict()
        self.children = OrderedDict()

    def find_genes(self):
        for ind in self.population:
            for e in ind:
                if e not in self.genes:
                    self.genes.append(e)
        if not self.init_genes:
            self.init_genes = self.genes
        if not self.term_genes:
            self.term_genes = self.genes

    def __save(self):
        """
        Logs state of the evolutionary optimization progress at each iteration
        :return: None
        """
        from pickle import dumps

        fl = open(self.check_point + self.task_name + ".eoa", "wb")
        info = dict(
            population_size=self.population_size,
            parents_porp=self.parents_porp,
            num_parents=self.num_parents,
            elits_porp=self.elits_porp,
            num_elites=self.num_elites,
            mutation_prob=self.mutation_prob,
            max_generations=self.max_generations,
            generation_num=self.generation_num,
            genes=self.genes,
            init_genes=self.init_genes,
            term_genes=self.term_genes,
            task_name=self.task_name,
            check_point=self.check_point,
            evals=self.evals,
            parents=self.parents,
            children=self.children,
        )
        fl.write(dumps(info))
        fl.close()

    def __load(self):
        """
        Loads previous information saved, if any
        :return: None
        """
        from pickle import loads

        try:
            fl = open(self.check_point + self.task_name + ".eoa", "rb")
            info = loads(fl.read())
            fl.close()
            self.population_size = info["population_size"]
            self.parents_porp = info["parents_porp"]
            self.num_parents = info["num_parents"]
            self.elits_porp = info["elits_porp"]
            self.num_elites = info["num_elites"]
            self.mutation_prob = info["mutation_prob"]
            self.max_generations = info["max_generations"]
            self.generation_num = info["generation_num"]
            self.genes = info["genes"]
            self.init_genes = info["init_genes"]
            self.term_genes = info["term_genes"]
            self.task_name = info["task_name"]
            self.check_point = info["check_point"]
            self.evals = info["evals"]
            self.parents = info["parents"]
            self.children = info["children"]
        except FileNotFoundError:
            pass

    def __call__(self, *args, **kwargs):
        self.parents = self.init_pop(self)
        self.__load()
        tqdm = None
        pbar = None
        try:
            ipy_str = str(type(get_ipython()))  # notebook environment
            if "zmqshell" in ipy_str:
                from tqdm import tqdm_notebook as tqdm
            if "terminal" in ipy_str:
                from tqdm import tqdm
        except NameError:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
        if tqdm is not None:
            pbar = tqdm(total=self.max_generations)
        pbar.update(self.generation_num)
        while not self.termination(self):
            self.__save()
            self.generation_num += 1
            self.parents = self.fitness(self.parents)
            for _ in self.parents:
                self.evals[_] = self.parents[_]
            self.recomb(self)
            self.mutation(self)
            self.children = self.fitness(self.children)
            for _ in self.children:
                self.evals[_] = self.children[_]
            self.elitism(self)
            self.parents = self.children
            if tqdm is not None:
                pbar.update(1)


class UniformRand(object):
    """
    Initial population initiation.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, ref, *args, **kwargs):
        from collections import OrderedDict
        from random import randint

        cnt = 0
        indices = []
        while cnt < ref.num_parents:
            idx = randint(0, ref.population_size - 1)
            if idx not in indices:
                # print(ref.population[idx])
                indices.append(idx)
                cnt += 1
        return OrderedDict(
            [(ref.population[i], ref.evals[ref.population[i]]) for i in indices]
        )


class MaxGenTermination(object):
    """
    Termination condition: Whether the maximum number of generations has been reached or not
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, ref, *args, **kwargs):
        if ref.generation_num < ref.max_generations:
            return False
        return True


class UniformCrossover(object):
    """
    Recombination procedure.
    """

    def __init__(self, **kwargs):
        self.fitnesses = []
        self.fmin = 0.0
        self.fmax = 0.0
        self.kwargs = kwargs
        self.mated = False

    def scale(self, scrs):
        self.fmin = min(scrs)
        self.fmax = max(scrs)
        lng = self.fmax - self.fmin
        if lng == 0.0:
            lng = 1.0
        self.fitnesses = [(_ - self.fmin) / lng for _ in scrs]

    def select_idx(self):
        from random import uniform

        fsum = sum(self.fitnesses)
        r = uniform(0.0, fsum)
        idx = 0
        F = self.fitnesses[idx]
        while F < r:
            idx += 1
            F += self.fitnesses[idx]
        return idx

    def pair(self):
        p1 = self.parents.pop(-1)
        self.fitnesses.pop(-1)
        idx = self.select_idx()
        p2 = self.parents.pop(idx)
        self.fitnesses.pop(idx)
        return p1, p2

    def mate(self, p1, p2):
        from random import randint

        l1 = len(p1)
        l2 = len(p2)
        if l1 == 1 and l2 == 1:
            c1 = (p1[0], p2[0])
            c2 = (p2[0], p1[0])
            return c1, c2
        r = randint(1, max(l1, l2))
        cl1 = max(0, l1 - r)
        cl2 = max(0, l2 - r)
        c1l = list(p1)[:cl1]
        c1r = list(p1)[cl1:]
        c2l = list(p2)[:cl2]
        c2r = list(p2)[cl2:]
        c1 = tuple(c1l + c2r)
        c2 = tuple(c2l + c1r)
        self.mated = True
        return c1, c2

    def __call__(self, ref, *args, **kwargs):
        from collections import OrderedDict

        ref.parents = OrderedDict(sorted(ref.parents.items(), key=lambda x: x[1]))
        self.parents = list(ref.parents.keys())
        self.scale(list(ref.parents.values()))
        children = []
        while len(self.parents) > 1:
            p1, p2 = self.pair()
            c1, c2 = self.mate(p1, p2)
            if c1 in ref.evals:
                children += [c1]
            if c2 in ref.evals:
                children += [c2]
        ref.children = OrderedDict([(c, ref.evals[c]) for c in children])
        ref.children = ref.fitness(ref.children)
        ref.children = OrderedDict(sorted(ref.children.items(), key=lambda x: x[1]))


class Elites(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, ref, *args, **kwargs):
        from collections import OrderedDict

        children = ref.children
        parents = ref.parents
        dif_ = ref.num_elites + len(parents) - len(children)
        top_elites = list(parents.items())[-dif_:]
        top_children = list(children.items())[dif_:]
        ref.children = OrderedDict(
            sorted(top_elites + top_children, key=lambda x: x[1])
        )


class Mutation(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, ref, *args, **kwargs):
        from random import uniform, randint
        from collections import OrderedDict

        mchildren = []
        for chld in ref.children:
            mchld = []
            idx = 0
            length = len(chld) - 1
            for e in chld:
                me = e
                prb = uniform(0, 1.0)
                if prb <= ref.mutation_prob:
                    if idx == 0:
                        lng = len(ref.init_genes)
                        rdx = randint(0, lng)
                        if rdx < lng:
                            me = ref.init_genes[rdx]
                        else:
                            me = ""
                    elif idx == length:
                        lng = len(ref.term_genes)
                        rdx = randint(0, lng)
                        if rdx < lng:
                            me = ref.term_genes[rdx]
                        else:
                            me = ""
                    else:
                        lng = len(ref.genes)
                        rdx = randint(0, lng)
                        if rdx < lng:
                            me = ref.genes[rdx]
                        else:
                            me = ""
                if me != "":
                    mchld.append(me)
                idx += 1
            new_chld = tuple(mchld)
            if new_chld in ref.evals:
                mchildren.append((new_chld, ref.evals[new_chld]))
        ref.children = OrderedDict(mchildren)


class Words(object):
    """
    This class takes a set as alphabet and generates words of a given length accordingly.
    A `Words` instant accepts the following parameters:

    :param letters: is a set of letters (symbols) to make up the words
    :param last: a subset of `letters` that are allowed to appear at the end of a word
    :param first: a set of words that can only appear at the beginning of a word
    :param repeat: whether consecutive occurrence of a letter is allowed
    """

    def __init__(self, letters, last=None, first=None, repeat=False):
        self.letters = letters
        self.last = last
        self.first = first
        self.words = []
        self.repeat = repeat

    def _check_cons_repeat(self, o):
        lng = len(o)
        for i in range(1, lng):
            if self.repeat:
                return True
            if o[i - 1] == o[i]:
                return False
        return True

    def _check_mid_first(self, o):
        if (self.first is None) or (self.first == []):
            return True
        n_ = len(o)
        for i in range(1, n_):
            if o[i] in self.first:
                return False
        return True

    def Generate(self, l):
        """
        Generates the set of legitimate words of length `l`
        :param l: int, the length of words
        :return: set of all legitimate words of length `l`
        """
        from itertools import product

        words = []
        for o in product(self.letters, repeat=l):
            if self.last is not None:
                if o[-1] in self.last:
                    if self._check_cons_repeat(o) and self._check_mid_first(o):
                        words.append(o)
            else:
                if self._check_cons_repeat(o) and self._check_mid_first(o):
                    words.append(o)
        return words
