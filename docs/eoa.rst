========================================
Evolutionary Optimization Algorithms
========================================
``SKSurrogate.eoa`` provides a small, configurable evolutionary optimizer for
discrete populations. It is also used by ``AML.eoa_fit`` to search over tuples
of pipeline components.

The optimizer works with a complete population supplied by the caller. Each
individual is normally a tuple, for example ``('scale', 'model')``. A fitness
function evaluates groups of individuals and returns an
``collections.OrderedDict`` mapping each individual to a numeric fitness.
Higher fitness values are treated as better by the built-in crossover and
elitism operators.

Algorithm
=========
For each generation, ``EOA`` performs the following steps:

1. Select an initial set of parents from the population.
2. Evaluate the parents with the fitness function.
3. Recombine parent pairs into children.
4. Mutate the children.
5. Evaluate the mutated children.
6. Preserve elite parents and use the resulting children as the next parents.
7. Save a checkpoint and continue until the termination operator returns
   ``True``.

The default operators are:

``UniformRand``
    Selects distinct initial parents uniformly from the population.
``UniformCrossover``
    Selects mating pairs using their fitness and exchanges tuple suffixes to
    create two children.
``Mutation``
    Replaces tuple elements with probability ``mutation_prob``.
``Elites``
    Carries the best parents into the next generation.
``MaxGenTermination``
    Stops when ``generation_num`` reaches ``max_generation``.

Quick start
===========
This complete example can be run from the repository root with::

    mkdir -p eoa-checkpoints
    .venv/bin/python - <<'PY'
    from collections import OrderedDict

    from SKSurrogate.eoa import EOA

    from itertools import product

    population = [
        tuple(word)
        for length in range(1, 5)
        for word in product("abcd", repeat=length)
    ]

    def fitness(individuals):
        # EOA maximizes fitness. This example prefers later letters.
        return OrderedDict(
            (individual, sum(ord(letter) for letter in individual))
            for individual in individuals
        )

    optimizer = EOA(
        population=population,
        fitness=fitness,
        num_parents=4,
        mutation_prob=0.1,
        max_generation=5,
        task_name="letters",
        check_point="./eoa-checkpoints/",
    )
    optimizer()

    best = max(
        ((individual, score) for individual, score in optimizer.evals.items()
         if score is not None),
        key=lambda item: item[1],
    )
    print(best)
    PY

The optimizer does not create the checkpoint directory automatically.

Fitness callback contract
=========================
The callback receives an ``OrderedDict`` whose keys are individuals. It must
return an ``OrderedDict`` with the same shape and numeric values::

    def fitness(individuals):
        return OrderedDict(
            (individual, evaluate(individual))
            for individual in individuals
        )

The callback may receive individuals whose fitness was previously stored in
``optimizer.evals``. Reusing those values is optional, but can avoid expensive
objective evaluations. Fitness values should be comparable with ``<`` and
``>``; the built-in operators assume larger values are better.

Constructor parameters
======================
``EOA(population, fitness, **kwargs)`` accepts these keyword arguments:

``population``
    Complete iterable of possible individuals. ``UniformRand`` requires at
    least ``num_parents`` individuals.
``fitness``
    Callback described above.
``init_pop``
    Class used to select initial parents. Defaults to ``UniformRand``.
``recomb``
    Class used to create children. Defaults to ``UniformCrossover``.
``mutation``
    Class used to mutate children. Defaults to ``Mutation``.
``termination``
    Class used to stop the run. Defaults to ``MaxGenTermination``.
``elitism``
    Class used to preserve strong parents. Defaults to ``Elites``.
``num_parents``
    Number of parents. If omitted, it is derived from ``parents_porp``.
``parents_porp``
    Parent proportion used when ``num_parents`` is omitted. The implementation
    default is ``0.1``.
``elits_porp``
    Proportion used to calculate the number of elites. The default is ``0.2``.
``mutation_prob``
    Probability of changing each child element. The default is ``0.05``.
``max_generation``
    Maximum generation count for the default termination operator. The default
    is ``50``. The keyword is singular: ``max_generation``.
``genes``
    Complete gene list used by mutation. If omitted, it is inferred from the
    population.
``init_genes``
    Genes allowed in the first tuple position during mutation.
``term_genes``
    Genes allowed in the last tuple position during mutation.
``task_name``
    Checkpoint filename prefix. The default is ``EOA``.
``check_point``
    Directory prefix for checkpoints. The file is formed as
    ``check_point + task_name + '.eoa'`` and the directory must already exist.

Custom operators
================
Custom operators are classes instantiated by ``EOA`` and called with the
``EOA`` instance. They can inspect or modify its state:

``init_pop``
    ``__call__(eoa)`` returns an ``OrderedDict`` of initial parents.
``recomb``
    ``__call__(eoa)`` sets ``eoa.children`` to an ``OrderedDict`` of children.
``mutation``
    ``__call__(eoa)`` modifies ``eoa.children`` after recombination.
``termination``
    ``__call__(eoa)`` returns ``True`` when optimization should stop.
``elitism``
    ``__call__(eoa)`` modifies ``eoa.children`` to include preserved parents.

For custom operators, use ``eoa.parents``, ``eoa.children`` and
``eoa.evals`` rather than maintaining a second population state. The main
loop evaluates parents and mutated children through the supplied fitness
callback.

Pipeline populations
====================
``AML`` uses ``aml.Words`` to create tuple populations. ``Words.Generate``
produces all tuples of a requested length while enforcing its ``first``,
``last`` and ``repeat`` rules. A valid AML population should contain at least
as many individuals as the requested ``num_parents`` and should have terminal
individuals that end in a classifier or regressor.

Checkpointing and resuming
==========================
EOA writes a pickle checkpoint before each generation. Use a separate
``task_name`` or directory for independent runs; two concurrent optimizers
must not share the same checkpoint path. Checkpoint files contain internal
optimizer state and should be treated as generated artifacts. The repository
``.gitignore`` excludes ``*.eoa`` files.

The checkpoint is loaded after the initial parent selection. To resume a run,
construct ``EOA`` with the same population, operator configuration,
``task_name`` and ``check_point`` values, then call it again.
