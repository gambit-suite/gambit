Tutorial
********


Before starting, make sure you have followed the instructions in :ref:`install-page`.

Also see the :ref:`cli-page` page for complete documentation of all GAMBIT subcommands
and options.


Example data set
================

The examples in this page make use of the following genome assembly files:

* `16AC1611138-CAP.fasta.gz`_
* `17AC0001410A.fasta.gz`_
* `17AC0006310.fasta.gz`_
* `17AC0006313-1.fasta.gz`_
* `19AC0011210.fasta.gz`_

From here on we'll pretend they've been downloaded to a directory called ``genomes/``.

These are from genome set 3 in the initial GAMBIT publication, derived from clinical samples.

.. _16AC1611138-CAP.fasta.gz: https://storage.googleapis.com/hesslab-gambit-public/genomes/set3/fasta/16AC1611138-CAP.fasta.gz>
.. _17AC0001410A.fasta.gz: https://storage.googleapis.com/hesslab-gambit-public/genomes/set3/fasta/17AC0001410A.fasta.gz>
.. _17AC0006310.fasta.gz: https://storage.googleapis.com/hesslab-gambit-public/genomes/set3/fasta/17AC0006310.fasta.gz>
.. _17AC0006313-1.fasta.gz: https://storage.googleapis.com/hesslab-gambit-public/genomes/set3/fasta/17AC0006313-1.fasta.gz>
.. _19AC0011210.fasta.gz: https://storage.googleapis.com/hesslab-gambit-public/genomes/set3/fasta/19AC0011210.fasta.gz>


.. _Locate DB:

Telling GAMBIT where the database files are
===========================================

The :ref:`query <query-cmd>` command requires the GAMBIT reference database files. You can let GAMBIT
know which directory contains these files in one of two ways:

Using a command line option
---------------------------

The first is to explicitly pass it via the command line using the ``--db`` option, like so::

    gambit --db /path/to/database/ COMMAND ...

Note that this option must appear immediately after ``gambit`` and before the command name.

Using an environment variable
-----------------------------

The second is to use the ``GAMBIT_DB_PATH`` environment variable. This can be done by
running the following command at the beginning of your shell session::

    export GAMBIT_DB_PATH="/path/to/database/"

Alternatively, you can add this line to your ``.bashrc`` to have it apply to all future sessions
(make sure to restart your current session after doing so).


Genome input
============

Genome assemblies used as input must be in FASTA format, optionally compressed with gzip.

Most commands accept a list of genome files as positional arguments, e.g.::

    gambit COMMAND [OPTIONS] genomes/16AC1611138-CAP.fasta.gz genomes/17AC0001410A.fasta.gz ...

or making use of shell expansion::

    gambit COMMAND [OPTIONS] genomes/*.fasta.gz

Alternatively you can use the ``-l`` option to provide a text file containing the genome file
names/paths, one per line. The paths in this file are considered relative to the directory given by
the ``--ldir`` option if given.

So for example, you can create the file ``genomes.txt`` containing the following::

    16AC1611138-CAP.fasta.gz
    17AC0001410A.fasta.gz
    17AC0006310.fasta.gz
    17AC0006313-1.fasta.gz
    19AC0011210.fasta.gz

The command would then be::

    gambit COMMAND [OPTIONS] -l genomes.txt --ldir genomes/

This method makes more sense when you have a lot of files to include.
Note that the ``gambit dist`` command has different names for these options because there are two
lists of genomes to specify. See the :ref:`cli-page` page for more complete information.


Predicting taxonomy of unknown genomes
======================================

The :ref:`query <query-cmd>` command compares a set of query genomes against the reference database
and attempts to predict their taxonomy. The following runs a query with our five FASTA files and
writes the results to ``out.csv``::

    gambit query -o out.csv genomes/*.fasta.gz


.. csv-table:: Contents of out.csv
   :file: example-results.csv
   :header-rows: 1

The ``predicted`` columns describe the predicted taxonomic classification of each query genome.
``closest.description`` is the database reference genome closest to the query, ``closest.distance``
is the distance between them.
The ``next`` columns have the same format as ``predicted`` but describe the next most specific taxon
for which the classification threshold was not met.

In this example GAMBIT was able to make a species-level prediction for the first three genomes
but stopped at the genus level for the fourth and made no prediction for the fifth.
This is because GAMBIT attempts to be conservative and error on the side of making a less specific
prediction or no prediction rather than giving false positives. The ``next`` columns can give you a
clue as to what a more specific classifiction might be, however.

See the :ref:`cli documentation <query-result-formats>` for a complete description of the output
columns. Generally the CSV output format should be sufficient, but there is also a JSON-based format
which contains more detailed information and may be useful in pipelines. Use ``-f json`` to use this
format.

.. todo::

    Explain why ``predicted.threshold`` is sometimes zero for certain taxa.


Pre-computing k-mer signatures
==============================

 TODO



Calculating GAMBIT distances
============================

TODO


Creating relatedness trees
==========================

TODO

