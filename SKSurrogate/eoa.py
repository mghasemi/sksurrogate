"""Configurable evolutionary optimization operators.

The :class:`EOA` coordinator evolves a supplied population of tuple-like
individuals. Fitness values are maximized by the built-in crossover and
elitism operators.
"""


class EOA(object):
    """Coordinate population initialization, evolution, and checkpointing.

    ``fitness`` receives an ``OrderedDict`` of individuals and must return an
    ``OrderedDict`` with numeric fitness values. Larger values are considered
    better. Operators are classes; they are instantiated with the remaining
    keyword arguments and called with this ``EOA`` instance.

    :param population: Complete collection of possible individuals.
    :param fitness: Callback that evaluates an ``OrderedDict`` of individuals.
    :param init_pop: Parent initializer class; defaults to ``UniformRand``.
    :param recomb: Recombination class; defaults to ``UniformCrossover``.
    :param mutation: Mutation class; defaults to ``Mutation``.
    :param termination: Termination class; defaults to ``MaxGenTermination``.
    :param elitism: Elitism class; defaults to ``Elites``.
    :param num_parents: Number of parents selected each generation. If omitted,
        it is derived from ``parents_porp``.
    :param parents_porp: Parent proportion used when ``num_parents`` is omitted;
        defaults to ``0.1``.
    :param elits_porp: Proportion used to calculate the elite count; defaults to
        ``0.2``.
    :param mutation_prob: Probability of changing each individual element;
        defaults to ``0.05``.
    :param max_generation: Maximum generation count for the default termination
        operator; defaults to ``50``.
    :param genes: Complete gene list used by ``Mutation``. Inferred from the
        population when omitted.
    :param init_genes: Genes allowed in the first position during mutation.
    :param term_genes: Genes allowed in the last position during mutation.
    :param task_name: Checkpoint filename prefix; defaults to ``"EOA"``.
    :param check_point: Directory prefix for checkpoints; defaults to ``"./"``.
        The directory must already exist.
    :param random_state: Optional seed for reproducible built-in operators.
    """
    CHECKPOINT_VERSION = 1


    def __init__(self, population, fitness, **kwargs):

        from collections import OrderedDict
        import random

        self.population = population
        self.random_state = kwargs.pop("random_state", None)
        self.rng = random.Random(self.random_state)
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
        """Infer mutation genes and positional gene lists from the population."""
        for ind in self.population:
            for e in ind:
                if e not in self.genes:
                    self.genes.append(e)
        if not self.init_genes:
            self.init_genes = self.genes
        if not self.term_genes:
            self.term_genes = self.genes

    def __save(self):
        """Serialize the current optimizer state to the ``.eoa`` checkpoint."""
        from pickle import dumps

        fl = open(self.check_point + self.task_name + ".eoa", "wb")
        info = dict(
            checkpoint_version=self.CHECKPOINT_VERSION,
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
            random_state=self.random_state,
            rng_state=self.rng.getstate(),
            evals=self.evals,
            parents=self.parents,
            children=self.children,
        )
        fl.write(dumps(info))
        fl.close()

    def __load(self):
        """Restore a matching checkpoint when one exists."""
        from pickle import loads

        try:
            fl = open(self.check_point + self.task_name + ".eoa", "rb")
            info = loads(fl.read())
            fl.close()
            if info.get("checkpoint_version", self.CHECKPOINT_VERSION) != self.CHECKPOINT_VERSION:
                raise ValueError("Unsupported EOA checkpoint version")
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
            self.random_state = info.get("random_state", self.random_state)
            if "rng_state" in info:
                self.rng.setstate(info["rng_state"])
            self.evals = info["evals"]
            self.parents = info["parents"]
            self.children = info["children"]
        except FileNotFoundError:
            pass

    def __call__(self, *args, **kwargs):
        """Run generations until the configured termination operator returns true.

        The method mutates ``parents``, ``children``, ``evals`` and
        ``generation_num`` in place. It returns ``None``; inspect ``evals`` or
        ``children`` after the run to retrieve results.
        """
        self.parents = self.init_pop(self)
        self.__load()
        #tqdm = None
        #pbar = None
        #try:
        #    from tqdm import tqdm
        #except ImportError:
        #    tqdm = None
        #if tqdm is not None:
        #    pbar = tqdm(total=self.max_generations)
        #pbar.update(self.generation_num)
        while not self.termination(self):
            print("Generation {g} of {t}".format(g=self.generation_num, t=self.max_generations))
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
            #if tqdm is not None:
            #    pbar.update(1)


class UniformRand(object):
    """Select distinct initial parents uniformly from the population."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, ref, *args, **kwargs):
        """Return an ``OrderedDict`` containing ``ref.num_parents`` individuals."""
        from collections import OrderedDict
        import random

        rng = getattr(ref, "rng", random)
        indices = rng.sample(range(ref.population_size), ref.num_parents)
        return OrderedDict(
            [(ref.population[i], ref.evals[ref.population[i]]) for i in indices]
        )


class MaxGenTermination(object):
    """Stop when ``ref.generation_num`` reaches ``ref.max_generations``."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, ref, *args, **kwargs):
        """Return ``True`` when the configured generation limit is reached."""
        if ref.generation_num < ref.max_generations:
            return False
        return True


class UniformCrossover(object):
    """Select fitness-weighted parent pairs and create tuple children.

    The operator expects numeric parent fitness values and writes the
    resulting children to ``ref.children``. Higher fitness receives greater
    selection weight.
    """

    def __init__(self, **kwargs):
        self.fitnesses = []
        self.fmin = 0.0
        self.fmax = 0.0
        self.kwargs = kwargs
        self.mated = False

    def scale(self, scrs):
        """Normalize fitness values into non-negative mating weights."""
        self.fmin = min(scrs)
        self.fmax = max(scrs)
        lng = self.fmax - self.fmin
        if lng == 0.0:
            lng = 1.0
        self.fitnesses = [(_ - self.fmin) / lng for _ in scrs]

    def select_idx(self):
        """Select a parent index using the current mating weights."""
        fsum = sum(self.fitnesses)
        r = self.rng.uniform(0.0, fsum)
        idx = 0
        F = self.fitnesses[idx]
        while F < r:
            idx += 1
            F += self.fitnesses[idx]
        return idx

    def pair(self):
        """Remove and return one fitness-weighted pair from the parent pool."""
        p1 = self.parents.pop(-1)
        self.fitnesses.pop(-1)
        idx = self.select_idx()
        p2 = self.parents.pop(idx)
        self.fitnesses.pop(idx)
        return p1, p2

    def mate(self, p1, p2):
        """Exchange tuple suffixes from two parents and return two children."""
        l1 = len(p1)
        l2 = len(p2)
        if l1 == 1 and l2 == 1:
            c1 = (p1[0], p2[0])
            c2 = (p2[0], p1[0])
            return c1, c2
        r = self.rng.randint(1, max(l1, l2))
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
        """Populate ``ref.children`` with recombined individuals."""
        from collections import OrderedDict

        self.rng = ref.rng
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
    """Preserve the highest-fitness parents among the next generation."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, ref, *args, **kwargs):
        """Merge elite parents into ``ref.children`` in fitness order."""
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
    """Mutate child genes using the configured per-element probability."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, ref, *args, **kwargs):
        """Replace genes in ``ref.children`` and remove empty mutations."""
        from collections import OrderedDict

        rng = ref.rng
        mchildren = []
        for chld in ref.children:
            mchld = []
            idx = 0
            length = len(chld) - 1
            for e in chld:
                me = e
                prb = rng.uniform(0, 1.0)
                if prb <= ref.mutation_prob:
                    if idx == 0:
                        lng = len(ref.init_genes)
                        rdx = rng.randint(0, lng)
                        if rdx < lng:
                            me = ref.init_genes[rdx]
                        else:
                            me = ""
                    elif idx == length:
                        lng = len(ref.term_genes)
                        rdx = rng.randint(0, lng)
                        if rdx < lng:
                            me = ref.term_genes[rdx]
                        else:
                            me = ""
                    else:
                        lng = len(ref.genes)
                        rdx = rng.randint(0, lng)
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
