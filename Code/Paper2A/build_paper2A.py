#!/usr/bin/env python3
"""
build_paper2A.py -- the methodological paper.

Target venue: Data Mining and Knowledge Discovery (Springer), subscription route.

Every quantity in the prose is a placeholder resolved from results/summary.json
or a results/table*.csv at build time.  Nothing is typed by hand.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import literature
from paperkit import Numbers, Paper, table_csv, fmt, RES, ROOT
from refs import Bib

OUT = ROOT / "Paper2A" / "Paper2A_leakage_premium.docx"


def build() -> Path:
    N = Numbers()
    B = Bib()
    d = Paper(N)

    # =====================================================================
    d.title("Leakage makes every problem look equally easy: "
            "a multi-dataset measurement of the resampling-before-splitting artefact")
    d.authors("Author names withheld for review",
              "Affiliations withheld for review")

    # ---------------------------------------------------------------- abstract
    d.abstract(
        f"""Applying synthetic over-sampling before rather than after the train-test split is
        among the most frequently reported causes of irreproducible results in applied machine
        learning. The error is well documented qualitatively; its magnitude has never been
        measured systematically. We measure it on {{n_datasets}} public binary classification
        datasets spanning imbalance ratios from {{ir_min}} to {{ir_max}}, with seven classifier
        families, four over-sampling methods and ten random seeds, holding every element of the
        pipeline fixed except the position of the resampler relative to the split. The mean
        inflation of minority-class F1 is {{mean_premium:.3f}}
        (95% CI {{premium_ci_lo:.3f}} to {{premium_ci_hi:.3f}}). More consequentially, leakage
        does not simply raise scores: it destroys the relationship between score and problem
        difficulty. Under the correct ordering, F1 varies with imbalance, dimensionality and
        class overlap as expected; under the leaked ordering it collapses onto a narrow band
        set mainly by the classifier's capacity to memorise interpolated near-duplicates, with
        {{variance_ratio}} times less variance. The premium is therefore almost entirely
        predicted by how hard the problem actually is
        (R-squared = {{law_r2}}). The leaked score is not merely insensitive to difficulty; it
        moves against it, rising with the imbalance ratio (r = {{corr_leaked_logir}}) exactly
        where the honest score falls (r = {{corr_correct_logir}}), because a more imbalanced problem
        requires more synthetic points and so puts more of the test set into training in
        interpolated form. A controlled simulation separates the factors that co-vary in real
        data and confirms the mechanism, which we trace to a measurable contamination rate. Two
        practical results follow. Over-sampling to a minority prevalence of 0.25 rather than to
        full balance cuts the artefact by {{mitigation_reduction_pct}}% and lowers the
        contamination that produces it, so partial resampling is a cheap partial safeguard. And because the artefact has a predictable signature, a
        reported result can be screened without access to the authors' code: a diagnostic trained
        on our corpus and validated leave-one-dataset-out separates leaked from honest results
        with AUC {{diag_auc}} using only the reported F1, imbalance ratio, minority count,
        feature count and model family. We release it as an open-source tool.""")

    d.keywords("data leakage; class imbalance; SMOTE; synthetic over-sampling; "
               "empirical evaluation; reproducibility; benchmark study")

    # =====================================================================
    d.h1("1  Introduction")

    d.p(f"""A machine learning result is a claim about a procedure applied to data that the
        procedure has not seen. Data leakage is the failure of that condition, and it is now
        recognised as a leading cause of results that do not replicate {B.c('kaufman2012',
        'kapoor2023', 'whalen2022')}. Among its many forms, one is unusually easy to commit and
        unusually easy to overlook: applying a synthetic over-sampler such as SMOTE
        {B.c('chawla2002')} to a dataset and only then splitting it into training and test
        partitions. Two lines of code in the wrong order produce a pipeline that looks correct,
        runs without warning, and reports a number that describes something other than what the
        author believes it describes.""")

    d.p("""The mechanism is not subtle once stated. SMOTE creates a new minority point by
        interpolating between a real minority point and one of its neighbours. If the
        interpolation happens before the split, some of those synthetic points inherit most of
        their coordinates from a real point that the split then assigns to the test set, while
        the synthetic point itself lands in the training set. The model does not have to
        generalise to those test cases. It has to remember them.""")

    d.p(f"""That the error inflates results has been stated many times, and demonstrated once at
        length on a single medical corpus {B.c('vandewiele2021')}. What has not been established
        is how large the inflation is, what governs its size, or whether it behaves consistently
        enough to be recognised in results one did not produce. Those questions matter for
        practical reasons. A reviewer confronted with a reported F1 of 0.95 on a problem with
        two hundred minority cases has no way to act on a qualitative warning. An editor deciding
        whether to request code has nothing to weigh. A field trying to understand why its
        published numbers cluster in a suspiciously narrow range has no quantitative account of
        why that would happen.""")

    d.p(f"""This paper supplies the measurement. We assembled {{n_datasets}} public binary
        classification problems, {{n_benchmark}} drawn from a standard benchmark collection
        {B.c('olson2017', 'romano2022')} and {{n_hr}} from the employee-attrition literature,
        with imbalance ratios from {{ir_min}} to {{ir_max}}, sample sizes from {{n_min_rows}} to
        {{n_max_rows}} and between {{p_min}} and {{p_max}} predictors. On each we fitted the same
        model three ways, changing nothing but where the resampler sits: before the split, after
        the split on the training partition only, and not at all. Seven classifier families, four
        over-samplers and ten seeds give {{n_paired}} matched leaked-versus-honest comparisons
        from {{n_results}} fitted models.""")

    d.p("""The headline number is the smaller half of the finding. The larger half is a
        regularity we did not expect and have not found reported: leakage does not shift the
        score distribution upward so much as it flattens it. Under the correct ordering, F1
        behaves as a measurement should, falling as the problem gets harder. Under the leaked
        ordering it stops responding to difficulty almost entirely, landing in a narrow band
        whose position depends mainly on which classifier was used. An easy problem and a hard
        problem, run through the same faulty pipeline, come out looking alike.""")

    d.p("""This has a consequence worth stating plainly, because it inverts the usual intuition.
        The common description of leakage is that it makes results 'a bit too good'. That
        description is wrong in a way that matters: leakage does not add a bias to a
        measurement, it replaces the measurement with a different one. The reported quantity
        stops being a property of the problem and becomes a property of the model class.""")

    d.p("Three contributions follow.")
    d.bullets([
        """C1 - Measurement. We quantify the inflation caused by resampling before splitting
        across a corpus that spans two orders of magnitude in imbalance, six to seven algorithm
        families and four over-sampling methods, and we report it with confidence intervals
        rather than point estimates.""",
        """C2 - A regularity and its mechanism. We show that the leaked score is close to
        invariant to problem difficulty and, where it does respond, responds in the wrong
        direction; that the premium is therefore predictable from difficulty alone; and that
        both follow from a measurable contamination rate. A
        controlled simulation separates imbalance, dimensionality, class overlap and sample
        size, which real data confounds.""",
        """C3 - A screening tool. Because the artefact has a stable signature, a published
        result can be screened from what papers already report. We train such a screen, validate
        it leave-one-dataset-out, and release it as an installable package.""",
    ])

    d.p(f"""A companion study by the present authors measured this effect on a single attrition
        benchmark and concluded that replication on independent data was needed before the
        finding could be generalised {B.c('paper1')}. The present work is that replication,
        at a scale that turns a case report into a measurement, and it asks the further
        questions the single-dataset design could not: whether the effect generalises, what
        governs its size, and whether it can be recognised from the outside.""")

    # =====================================================================
    d.h1("2  Related work")

    d.h2("2.1  Leakage")
    d.p(f"""Kaufman and colleagues {B.c('kaufman2012')} gave the first systematic treatment, defining leakage as the
        introduction of information about the target that would not be legitimately available at
        prediction time, and cataloguing the forms it takes. Two decades later, Kapoor and Narayanan
        {B.c('kapoor2023')} collated twenty-two review papers covering seventeen scientific
        fields and identified leakage affecting at least 294 published studies, arguing that it
        is a principal driver of the reproducibility problem in machine-learning-based
        science. Whalen and colleagues {B.c('whalen2022')} make the same case for genomics. The literature is rich in taxonomy and in individual case
        studies; it is thin in measurement. With the exception of Vandewiele and colleagues {B.c('vandewiele2021')}, who reproduced the inflated
        results of several preterm-birth studies and showed they
        collapsed under correct evaluation, we are not aware of work that quantifies the size of
        a specific leakage mechanism across many datasets.""")

    d.h2("2.2  Synthetic over-sampling and its sceptics")
    d.p(f"""SMOTE {B.c('chawla2002')} and its descendants Borderline-SMOTE {B.c('han2005')} and
        ADASYN {B.c('he2008')} are the most widely used response to class imbalance. A steady
        line of work questions whether they help. Blagus and Lusa {B.c('blagus2013')} showed that SMOTE is largely ineffective in high
        dimensions and can be harmful for some classifiers. Van den Goorbergh and colleagues
        {B.c('goorbergh2022')} demonstrated that imbalance corrections destroy the calibration
        of risk prediction models while leaving discrimination unchanged, a result Carriero and
        colleagues {B.c('carriero2025')} extended to machine learning models by simulation.
        Elor and Averbuch-Elor {B.c('elor2022')} benchmarked balancing against strong, properly tuned classifiers and
        found that it does not improve their performance at all, while confirming its utility
        for weaker learners - which is where earlier studies had demonstrated it. Our
        results are consistent with this line and add a specific observation: in our corpus the
        best-performing condition is frequently not to resample. Across the whole corpus, no
        resampling beat correctly-ordered resampling in {{pct_none_beats_correct}}% of
        matched comparisons.""")

    d.h2("2.3  Large-scale empirical benchmarking")
    d.p(f"""The practice of evaluating a claim across many datasets rather than a favoured few
        goes back to Demsar {B.c('demsar2006')}. PMLB {B.c('olson2017', 'romano2022')} curates a large
        collection of standardised classification and regression problems for exactly this
        purpose. Our design borrows the logic: the question is not whether the artefact appears
        in a particular case but how its magnitude behaves as a function of dataset properties,
        and that question needs a corpus wide enough to vary those properties.""")

    d.h2("2.4  The gap")
    d.p("""Three things are missing from the literature above. First, no measurement of the
        effect size of resampling-before-splitting as a function of dataset characteristics.
        Second, no account of what the leaked score is actually measuring when it is not
        measuring the problem. Third, no way for a reader to act on the concern in a specific
        case. This paper addresses all three.""")

    # =====================================================================
    d.h1("3  Method")

    d.h2("3.1  Corpus")
    d.p(f"""We assembled {{n_datasets}} binary classification problems in two groups. Group A
        consists of {{n_benchmark}} benchmark problems derived from PMLB
        {B.c('olson2017', 'romano2022')}. PMLB distributes datasets in a harmonised numeric
        format with a documented provenance; where a source is multi-class we binarise it
        one-versus-rest on the rarest class with at least thirty cases, which is the standard
        construction used to build imbalanced benchmark collections from multi-class sources.
        Group B consists of {{n_hr}} employee-attrition and turnover datasets, which supply a
        single applied domain in which the practice under study is common and which anchor the
        low end of the imbalance range (ratios {{hr_ir_min}} to {{hr_ir_max}}). One of them
        {B.c('alqahtani2025')} is survey data collected for the purpose rather than a
        re-publication of an existing corpus.""")

    d.p("""Two filters apply. A dataset enters only if it has at least thirty minority cases,
        because below that a 70:30 split leaves too few test positives for F1 to be stable.
        Datasets larger than 5,000 rows are stratified-subsampled to 5,000, which keeps the
        factorial tractable without changing any dataset's class balance. Both thresholds are
        recorded here because they are choices, not facts, and a reader may reasonably want to
        vary them; the released code takes both as parameters.""")

    d.table(
        fmt(table_csv("table1_corpus.csv"),
            keep=["name", "domain", "n", "n_features", "n_minority", "imbalance_ratio",
                  "construction"],
            dec={"imbalance_ratio": 2},
            rename={"name": "Dataset", "domain": "Domain", "n": "n",
                    "n_features": "p", "n_minority": "Minority",
                    "imbalance_ratio": "IR", "construction": "Construction"}),
        caption="""The corpus. IR is the ratio of majority to minority cases after any
        subsampling. 'Construction' records how a binary target was obtained: benchmark
        problems that were already binary are marked as such, and multi-class sources give the
        class that was taken as positive.""",
        fontsize=7.2, widths=[1.55, .72, .55, .45, .62, .5, 1.6])

    d.figure("fig01_corpus.png",
             """Coverage of the corpus. Each point is a dataset. The horizontal axis is the
             imbalance ratio and the vertical axis the number of minority cases, both on log
             scales. The employee-attrition datasets, labelled, occupy the mild-imbalance
             corner; the benchmark problems extend the range by two orders of magnitude, which
             is what allows the regularity in Section 4.2 to be stated as a function of
             difficulty rather than observed at a single point.""", width=4.6)

    d.h2("3.2  The three conditions")
    d.p("""Every comparison in this paper is between pipelines that differ in exactly one
        respect. The data, the preprocessing, the classifier, its hyperparameters, the resampler,
        its random seed, the split proportion and the split seed are all held fixed. The only
        thing that moves is where the resampler sits.""")

    d.table(pd.DataFrame([
        dict(Condition="leaked",
             Procedure="Resample the entire dataset, then split 70:30",
             Role="Reproduces the error as it appears in the literature"),
        dict(Condition="correct",
             Procedure="Split 70:30, then resample the training partition only",
             Role="The correct protocol"),
        dict(Condition="none",
             Procedure="Split 70:30, no resampling",
             Role="Baseline: is resampling helping at all?"),
    ]), caption="""The three conditions. Everything except the position of the resampler is
        held identical, so any difference between conditions is attributable to that position
        alone.""", fontsize=9, widths=[.9, 2.6, 2.5], align_right_from=99, numbered=False)

    d.p("""In the leaked condition the test partition is drawn from the resampled pool, so it
        contains synthetic points. This is deliberate: it is what the erroneous pipeline actually
        does, and reproducing it faithfully is the point. In the correct and none conditions the
        test partition contains only real observations. All features are standardised inside the
        pipeline, fitted on training data only in every condition, so that scaling is never the
        source of a difference.""")

    d.h2("3.3  A measure of the mechanism")
    d.p(f"""If the account in Section 1 is right, the leaked condition should be recognisable not
        only by its outcome but by an intermediate quantity: how much of the test set the
        training partition has effectively already seen. We define the contamination rate as the
        fraction of held-out minority points whose nearest synthetic training point lies closer
        than half the median nearest-neighbour distance among real minority points. The reference
        distance is computed on the same standardised scale, so the measure is dimensionless and
        comparable across datasets. It is the mediating variable that turns an observation into
        an explanation, and it is what lets us say why the mitigation in Section 4.6 works.""")

    d.h2("3.4  The factorial")
    d.table(pd.DataFrame([
        dict(Factor="Datasets", Levels=str(N.raw("n_datasets")),
             Detail=f"{N.raw('n_benchmark')} benchmark, {N.raw('n_hr')} HR/turnover"),
        dict(Factor="Classifiers", Levels="7",
             Detail="LogReg, DecisionTree, 5-NN, SVM-RBF, RandomForest, "
                    "GradientBoosting, XGBoost"),
        dict(Factor="Resamplers", Levels="4",
             Detail="SMOTE, Borderline-SMOTE, ADASYN, RandomOverSampler"),
        dict(Factor="Conditions", Levels="3", Detail="leaked / correct / none"),
        dict(Factor="Seeds", Levels="10",
             Detail="so that every comparison carries an interval, not a point"),
        dict(Factor="Model fits", Levels=f"{N.raw('n_fits_attempted'):,}",
             Detail=f"{N.raw('n_results'):,} completed; "
                    f"{N.raw('n_paired'):,} matched leaked-correct pairs"),
    ]), caption="""The design. The seed dimension exists so that every comparison in Section 4
        can be reported with a confidence interval.""",
        fontsize=9, widths=[1.15, .75, 4.1], align_right_from=99, numbered=False)

    d.p(f"""Classifiers are used at library defaults except where a default is plainly
        unsuitable for a study of this kind: the decision tree is capped at depth six and the
        forests at two hundred trees. We do not tune. Tuning would introduce a second source of
        variation between conditions and would raise the question of whether the tuning itself
        leaked. The absence of tuning means our absolute scores are not competitive with
        published ones; that is irrelevant here, because every number we report is a difference
        between two pipelines that share the same untuned model. Metrics recorded are
        minority-class F1, precision, recall, Matthews correlation, ROC-AUC and average
        precision; F1 is the headline because it is what the literature under study reports, and
        the others are used in Section 4.2 to check that the finding is not an artefact of one
        metric {B.c('saito2015', 'chicco2020')}.""")

    d.h2("3.5  Controlled simulation")
    d.p("""Real datasets confound the things we most want to separate. Highly imbalanced problems
        tend also to be larger, higher-dimensional and more overlapping, because that is how such
        problems arise. A synthetic grid pulls the factors apart: we generate problems at five
        imbalance ratios, three dimensionalities, three class-separation levels and two sample
        sizes, and run the leaked and correct conditions on each with five classifiers and three
        seeds. The simulation is a supplement to the observational corpus and not a substitute
        for it; its role is to establish which of the co-varying factors is doing the work, and
        it is presented after the empirical results for that reason.""")

    d.h2("3.6  Meta-regression")
    d.p("""To ask what governs the premium we regress each of the three quantities - the honest
        score, the leaked score and their difference - on the same set of descriptors: log
        imbalance ratio, log minority count, log feature count, log sample size, and indicator
        variables for classifier and resampler. Standard errors are clustered by dataset, since
        observations from one dataset are not independent. The comparison of interest is between
        the R-squared values: if the leaked score has stopped measuring the problem, the
        descriptors of the problem should explain much less of its variance than they explain of
        the honest score's.""")

    d.h2("3.7  The screening model and how it is validated")
    d.p("""Each matched pair yields two labelled examples: a leaked result and an honest one,
        identical in every respect a paper would report except the number itself. We train a
        gradient-boosted classifier to tell them apart using only features a reader can extract
        from a published paper - the reported F1, the imbalance ratio, the minority count, the
        feature count and the model family - and validate it leave-one-dataset-out, so that the
        estimate is never inflated by similarity between observations from the same dataset. A
        model that could only recognise the datasets it was trained on would be useless for the
        purpose, and leave-one-dataset-out is the validation that detects that failure.""")

    d.h2("3.8  Reproducibility")
    d.p(f"""All datasets are public. The corpus builder downloads Group A and reads Group B from
        files with documented provenance, then writes a manifest that every downstream step reads.
        The experiment checkpoints after each dataset. All analysis flows into a single
        machine-readable summary, and every quantity in this manuscript is inserted from that
        summary at build time rather than typed, so the text cannot drift from the tables.
        Library versions at the time of the reported run were Python {{versions.python}},
        NumPy {{versions.numpy}}, pandas {{versions.pandas}}, SciPy {{versions.scipy}},
        scikit-learn {{versions.scikit_learn}} {B.c('pedregosa2011')},
        imbalanced-learn {{versions.imbalanced_learn}} {B.c('lemaitre2017')} and
        XGBoost {{versions.xgboost}} {B.c('chen2016')}.""")

    # =====================================================================
    d.h1("4  Results")

    d.h2("4.1  The corpus behaves as intended")
    d.p("""Table 1 and Figure 1 describe what was assembled. The imbalance ratio spans
        {ir_min} to {ir_max}, minority counts {minority_min} to {minority_max}, and feature
        counts {p_min} to {p_max}. The employee-attrition datasets sit at the mild end of the
        imbalance range, which is worth noting for what follows: the domain in which this error
        is most often committed is also the domain in which the honest score is highest, and
        therefore the domain in which the premium is smallest. The corpus is not a random sample
        of problems, and no claim here depends on treating it as one.""")

    d.h2("4.2  The invariance")
    d.p("""This is the central result. Under the correct ordering, mean minority-class F1 is
        {correct_f1_mean} with a standard deviation of {correct_f1_sd}. Under the leaked
        ordering, the mean rises to {leaked_f1_mean} but the standard deviation falls to
        {leaked_f1_sd}. The honest score has {variance_ratio} times the variance of the leaked
        one. Aggregating first to dataset means, which removes within-dataset noise, the ratio
        is {variance_ratio_bydataset}: across the whole corpus, mean honest F1 ranges over
        {correct_range_bydataset} while mean leaked F1 ranges over only
        {leaked_range_bydataset}.""")

    d.p("""Figure 2(a) shows the shape of it. Points lie above the diagonal, as expected. What is
        not expected is that they lie along a horizontal band rather than along a line parallel
        to the diagonal. A pipeline that merely added a bias would produce the latter.""")

    d.p("""Figure 2(b) makes the same point against the variable that should matter most, and
        adds one we did not anticipate. As the imbalance ratio rises across the corpus, mean
        honest F1 falls from {ir_q1_correct} in the least imbalanced quartile to {ir_q4_correct}
        in the most imbalanced, which is what one expects of a measurement tracking a problem
        that is getting harder. Over the same range the leaked score does not merely stay flat:
        it rises, from {ir_q1_leaked} to {ir_q4_leaked}. The correlation between log imbalance
        ratio and F1 is {corr_correct_logir} under the correct ordering and {corr_leaked_logir}
        under the leaked one - opposite in sign, and larger in magnitude under leakage. Per
        decade of imbalance ratio the honest score falls by {slope_correct_logir_abs} while the
        leaked score rises by {slope_leaked_logir}.""")

    d.p("""The reason is the mechanism, and it belongs here rather than in Section 4.5. The more
        imbalanced a dataset is, the more synthetic points an over-sampler must manufacture to
        balance it, and so the more of the test set is present in the training partition in
        interpolated form. Difficulty and contamination rise together, and the second more than
        cancels the first. This is why the two scores diverge fastest exactly where the honest
        task is hardest, and it is what makes a high reported score on a highly imbalanced
        problem the specific thing to be suspicious of.""")

    d.figure("fig02_invariance.png",
             """The invariance. (a) Leaked against honest F1 for all {n_paired} matched pairs;
             the dashed line is equality and the horizontal line the mean leaked score. Points
             lie along a band, not along a shifted diagonal. (b) Both scores against the
             imbalance ratio, binned; error bars are one standard deviation. The honest score
             falls as the problem gets harder; the leaked score rises. (c) The premium against honest
             F1, with the fitted line: how much a result is inflated is almost entirely
             determined by how hard the problem actually was.""")

    d.p("""Figure 2(c) states the consequence as a relation. The premium is
        premium = {law_intercept:.3f} - {law_slope_abs:.3f} x F1(correct), with R-squared
        {law_r2}. Aggregated to dataset means the fit is {law_r2_bydataset}. Nearly all of the
        variation in how much a result is inflated is explained by how hard the problem was to
        begin with - which is another way of saying that the leaked score carries almost no
        information about the problem, so the difference between it and the honest score is
        almost entirely the honest score.""")

    d.p("""Table 2 checks the invariance against the design factors one at a time; within each
        factor we report the spread of condition means across levels. Across imbalance quartiles
        the honest score spreads {spread_correct_imbalance} and the leaked score
        {spread_leaked_imbalance}, in opposite directions. Across classifiers the two spread by
        comparable amounts, {spread_correct_classifier} and {spread_leaked_classifier}, but they
        rank the classifiers differently: the leaked score is highest for {leaked_clf_max_name}
        and lowest for {leaked_clf_min_name}, an ordering by capacity to memorise rather than by
        any property of the data. Across resamplers both scores spread by less than
        {spread_correct_resampler}, so which over-sampler is used is close to irrelevant to the
        artefact. What survives in the leaked score is variation from the estimator and from the
        amount of contamination the pipeline manufactured, and not from the question being
        asked.""")

    d.table(
        fmt(table_csv("table2_invariance.csv"),
            dec={"leaked": 4, "correct": 4, "none": 4, "premium": 4},
            rename={"factor": "Factor", "level": "Level", "n": "Pairs",
                    "leaked": "Leaked F1", "correct": "Correct F1",
                    "none": "No resampling", "premium": "Premium"}),
        caption="""Invariance across the design factors. Each row is the mean over all matched
        pairs at that level. The honest score varies strongly across imbalance levels and
        weakly across classifiers; the leaked score does the reverse. The 'no resampling'
        column is populated only for classifiers, where it is defined.""",
        fontsize=8, widths=[1.15, 1.35, .7, .82, .85, .95, .78])

    d.p("""The finding is not an artefact of F1. Table 3 repeats the comparison for five other
        metrics. Matthews correlation shows a premium of {premium_mcc} with a variance ratio of
        {variance_ratio_mcc}; average precision shows {premium_prauc} and
        {variance_ratio_prauc}; ROC-AUC, which is insensitive to the class ratio and therefore
        the most conservative test available, still shows {premium_rocauc} and
        {variance_ratio_rocauc}. The direction and the flattening survive the change of
        metric.""")

    d.table(
        fmt(table_csv("table6_metrics.csv"),
            dec={"leaked_mean": 4, "leaked_sd": 4, "correct_mean": 4, "correct_sd": 4,
                 "premium": 4, "variance_ratio": 2},
            rename={"metric": "Metric", "leaked_mean": "Leaked", "leaked_sd": "sd",
                    "correct_mean": "Correct", "correct_sd": "sd ", "premium": "Premium",
                    "variance_ratio": "Var ratio"}),
        caption="""The result is not specific to F1. 'Var ratio' is the variance of the honest
        score divided by the variance of the leaked score; values above one mean the leaked
        score is the flatter of the two.""", fontsize=8.5)

    d.h2("4.3  What the meta-regression shows, and what we expected instead")
    d.p("""Table 4 reports the three regressions described in Section 3.6, and the result is not
        the one we set out to find. We expected the descriptors of a problem to explain the
        honest score well and the leaked score poorly. They do the opposite: dataset and
        algorithm descriptors explain {metareg_r2_leaked} of the variance in the leaked score
        and only {metareg_r2_correct} of the variance in the honest one. Restricting the model to
        data descriptors alone - imbalance, minority count, dimensionality, sample size, with no
        algorithm terms - the gap is similar, {dataonly_r2_leaked} against {dataonly_r2_correct}.
        Algorithm terms alone explain little of either ({algonly_r2_leaked} and
        {algonly_r2_correct}).""")

    d.p("""We report this because it is what the data says, and because reading it correctly
        strengthens rather than weakens the argument. The resolution is in the sign. The leaked
        score is highly predictable from the imbalance ratio, but it moves the wrong way: its
        correlation with log imbalance is {corr_leaked_logir}, positive, while the honest score's
        is {corr_correct_logir}. What the descriptors predict, when they predict the leaked
        score, is not how hard the problem is but how much contamination the pipeline
        manufactured - and imbalance is the main driver of that, because a more imbalanced
        problem needs more synthetic points to balance it.""")

    d.p("""So the leaked score is not noise, and it is not a degraded copy of the honest score
        either. It is a well-behaved measurement of a different quantity. Predictability is
        therefore not evidence that a leaked number carries information about the problem; it
        carries information about the error. That is a sharper claim than the one we expected
        to make, and it is what makes the screen in Section 4.7 possible at all: a signature
        this systematic can be recognised.""")

    d.p("""One further reading of Table 4 is worth stating. That the honest score is only weakly
        explained ({metareg_r2_correct}) by imbalance, sample size, dimensionality and model
        family is itself a useful result for anyone designing a benchmark study. Most of what
        makes a real classification problem hard is not captured by the descriptors a paper
        reports - class overlap and feature quality above all - which is precisely why the
        controlled simulation in the next section is necessary.""")

    d.table(
        fmt(table_csv("table3_metaregression.csv"),
            dec={c: 4 for c in ["correct_coef", "correct_se", "correct_p", "leaked_coef",
                                "leaked_se", "leaked_p", "premium_coef", "premium_se",
                                "premium_p"]},
            keep=["term", "correct_coef", "correct_se", "leaked_coef", "leaked_se",
                  "premium_coef", "premium_se"],
            rename={"term": "Term", "correct_coef": "Correct b", "correct_se": "se",
                    "leaked_coef": "Leaked b", "leaked_se": "se ",
                    "premium_coef": "Premium b", "premium_se": "se  "}),
        caption="""Meta-regression of each outcome on dataset and algorithm descriptors, with
        standard errors clustered by dataset. R-squared is {metareg_r2_correct} for the honest
        score, {metareg_r2_leaked} for the leaked score and {metareg_r2_premium} for the
        premium. The coefficients on the data terms are the ones to compare across columns.""",
        fontsize=7.6)

    d.h2("4.4  The simulation isolates the cause")
    d.p("""The corpus cannot separate imbalance from the things that travel with it. The
        simulation can. Across {sim_cells} synthetic fits, the same asymmetry appears: honest F1
        has {sim_variance_ratio} times the variance of leaked F1. Varying the imbalance ratio
        alone, with dimensionality, class separation and sample size held fixed, honest F1 falls
        from {sim_ir2_correct} to {sim_irmax_correct} - a drop of
        {sim_range_correct_ir} - while leaked F1 moves by {sim_range_leaked_ir}. Varying
        dimensionality alone moves the honest score by {sim_range_correct_dim} and the leaked
        score by {sim_range_leaked_dim}. Varying class separation moves them by
        {sim_range_correct_sep} and {sim_range_leaked_sep}.""")

    d.p("""Figure 3 shows all three comparisons side by side. This settles the direction of the
        causal story. The premium grows with imbalance not
        because leakage helps more when data is scarce, but because the honest score has further
        to fall. The leaked score barely moves at all. That is a different claim from 'leakage
        inflates results', and it is the one the data supports.""")

    d.figure("fig03_simulation.png",
             """Controlled simulation. Each panel varies one factor with the others held fixed;
             error bars are one standard deviation across cells and seeds. The leaked curve is
             close to flat in all three panels while the honest curve responds to each factor,
             which is what identifies problem difficulty rather than leakage magnitude as the
             thing that moves.""")

    d.h2("4.5  The mechanism, measured")
    d.p("""The contamination rate defined in Section 3.3 averages {contam_mean} across the
        corpus and ranges from {contam_min} to {contam_max}. Its correlation with the log
        imbalance ratio is {corr_contam_logir}, which identifies it as the channel through which
        imbalance raises the leaked score rather than lowering it. It correlates with the
        premium at r = {corr_contam_premium} and with the leaked score at
        r = {corr_contam_leaked}.""")

    d.p("""Contamination is a genuine mediator but not a complete one, and it is worth being
        precise about that. Entering it alongside honest difficulty in a regression on the
        premium moves the coefficient on difficulty only from {mediation_total:.3f} to
        {mediation_direct:.3f}, about {mediation_share_pct}% of the total. That is what one
        should expect from the definition: the contamination rate measures how much of the test
        set has been copied into training, and not what a given model does with the copies. How
        much a copy is worth depends on the estimator's capacity to memorise, which is why the
        classifier remains the second source of variation in Table 2.""")

    d.p("""The resampler comparison is the sharpest test the design offers, and it separates the
        two halves of the mechanism cleanly. RandomOverSampler duplicates existing minority
        points rather than interpolating new ones, so under the leaked ordering it puts exact
        copies of test points into training rather than near copies. It produces by far the
        highest contamination of any resampler in the corpus, {contam_RandomOS} against
        {contam_Borderline} for Borderline-SMOTE, which is exactly what the definition predicts.
        Its premium, however, is not the largest: it is {premium_RandomOS}, against
        {premium_ADASYN} for ADASYN. More contamination does not mean a proportionally larger
        artefact, because a duplicated point teaches a model less than an interpolated
        neighbourhood does. The four resamplers differ from one another by less than
        {spread_leaked_resampler} in leaked F1 - which is the substantive point. What matters is
        that synthetic points cross the split at all, not how they were made.""")

    d.h2("4.6  Partial resampling as a safeguard")
    d.p("""If the artefact comes from synthetic points crossing the split, resampling to less
        than full balance should shrink it, because there are fewer such points to cross. Table 5
        and Figure 4(b) confirm the direction across the whole corpus, and quantify a reduction
        that is real but partial. At full balance the premium is {premium_at_100:.3f}. At a
        target minority prevalence of 0.25 it is {premium_at_025:.3f}, a reduction of
        {mitigation_reduction_pct}%, and at 0.15 it is {premium_at_015:.3f}. The
        contamination rate falls in step, from {contam_at_100} to {contam_at_015}, which is what
        the mechanism predicts and what makes the reduction interpretable rather than merely
        convenient.""")

    d.p("""A quarter is not most of it. Partial resampling attenuates the artefact; it does not
        remove it, and a study that resamples to 0.15 before splitting still reports a number
        inflated by {premium_at_015:.2f}. Anyone hoping for a setting that makes the ordering
        safe should read Table 5 as ruling that out.""")

    d.p(f"""We are not recommending partial resampling as a way to make the error safe. The
        error should be fixed, not attenuated. The finding matters for a different reason: it
        gives a diagnostic signature. A pipeline whose reported performance changes sharply with
        the resampling ratio, in a way that cannot be explained by the classifier, is a pipeline
        worth inspecting. It is also a reason to doubt full rebalancing on its own merits, which
        several authors have argued from other directions
        {B.c('goorbergh2022', 'elor2022', 'carriero2025')}.""")

    d.table(
        fmt(table_csv("table4_mitigation.csv"),
            dec={"premium": 4, "premium_sd": 4, "leaked": 4, "correct": 4,
                 "contamination": 4, "lo": 4, "hi": 4},
            rename={"sampling_strategy": "Target prevalence", "n": "Fits",
                    "premium": "Premium", "premium_sd": "sd", "leaked": "Leaked",
                    "correct": "Correct", "contamination": "Contamination",
                    "lo": "CI low", "hi": "CI high"}),
        caption="""Premium as a function of how far SMOTE is asked to rebalance the data,
        across the whole corpus. Confidence intervals are bootstrap percentile intervals over
        matched fits.""", fontsize=8)

    d.figure("fig04_mechanism_mitigation.png",
             """(a) The premium against the contamination rate, binned into deciles; the
             mechanism is measurable, not merely plausible. (b) The premium and the
             contamination rate as functions of the target minority prevalence. Both fall
             together, which is the signature of a single cause.""", width=5.6)

    d.h2("4.7  Screening a reported result")
    d.p("""The final result asks whether the signature is stable enough to be recognised from
        outside. Using only the reported F1, imbalance ratio, minority count, feature count and
        model family, and validating leave-one-dataset-out over {diag_n} labelled results, the
        screen separates leaked from honest results with AUC {diag_auc}. Accuracy at the
        natural threshold is {diag_acc} and the Brier score is {diag_brier}.""")

    d.p("""The gap between two ways of scoring the same screen is instructive. Within a dataset,
        where the task is to say which of two results on the same problem was the leaked one,
        per-dataset AUC has median {diag_auc_perdataset_median} and exceeds 0.9 on
        {diag_n_datasets_above_09} of {diag_n_datasets_scored} datasets; it falls below chance
        on {diag_n_datasets_below_chance}. Pooled across datasets, where the task is to judge one
        reported result in isolation without knowing what an honest result on that problem would
        have looked like, the AUC is the {diag_auc} quoted above. The screen is therefore very
        good at the easy version of the question and moderately good at the hard one - and the
        hard one is what a reviewer actually faces. We report the pooled figure as the headline
        for that reason.""")

    d.p("""An ablation makes the interpretation concrete. Removing the reported F1 and leaving
        only the dataset descriptors drops the AUC to {diag_auc_without_f1}, close to chance, as
        it must be - the descriptors alone carry no information about which condition produced a
        result. The signal is entirely in the relationship between the reported score and the
        difficulty of the problem it was reported on, which is the relationship Section 4.2
        showed leakage destroys.""")

    d.figure("fig05_diagnostic.png",
             """(a) Leave-one-dataset-out ROC for the screen. (b) The screen as a reader would
             use it: probability that a reported result came from the leaked ordering, as a
             function of the reported F1, at four imbalance ratios, holding the other inputs at
             corpus medians. The curves shift left as the problem gets harder: on a hard
             problem, a moderately high score is already unusual.""", width=5.8)

    d.p("""We stress what the screen is and is not. It returns a probability that a number of
        that size, on a problem of that difficulty, came from the leaked ordering. It is not
        evidence of error in any individual paper, and it cannot be, because it sees only what
        the paper reported. Its use is to tell a reviewer or an editor when it is worth asking
        the authors a specific question: where does the resampler sit relative to the split?""")

    d.h2("4.8  What the published literature reports")
    lit = literature.frame()
    d.p(f"""Table 6 places our measured bands against results reported in the recent
        employee-attrition literature, a domain where over-sampling is near-universal and where
        the IBM HR benchmark gives an unusually clean comparison because so many studies use the
        same data. We classify each study only by what it says about its own protocol - never by
        what our screen guesses - into three groups: resampling confined to training data,
        resampling applied before splitting, and no statement either way.""")

    d.p(f"""The pattern is stark. The {int((lit.protocol=='train-only').sum())} studies that
        state resampling was confined to the training data report minority-class F1 between
        {lit[lit.protocol=='train-only'].f1.min():.3f} and
        {lit[lit.protocol=='train-only'].f1.max():.3f}
        {B.c('alyousef2026', 'cavescu2026')}. The study that states resampling preceded the
        split reports {float(lit[lit.protocol=='before-split'].f1.iloc[0]):.4f}
        {B.c('baydili2025')}. The {int((lit.protocol=='not stated').sum())} results that do not
        state the ordering report between {lit[lit.protocol=='not stated'].f1.min():.3f} and
        {lit[lit.protocol=='not stated'].f1.max():.3f}
        {B.c('nassreddine2026', 'alali2026', 'hamja2025')}. Our own measurements on the same
        dataset, where the ordering is known because we controlled it, put honest F1 at
        {{hr_by_dataset.ibm_hr.correct}} and leaked F1 at {{hr_by_dataset.ibm_hr.leaked}}.""")

    d.p("""We draw one conclusion from this and decline to draw another. The conclusion we draw
        is that the ordering is frequently not reported, and that where it is reported the
        difference in outcomes is large enough that a reader cannot treat the omission as
        immaterial. The conclusion we decline to draw is that any particular unreported study is
        in error. Several other explanations are available - different preprocessing, different
        splits, different metric definitions, a macro-averaged rather than minority-class F1 -
        and none of them can be ruled out from the outside. That is the point: from the outside,
        nothing can be ruled out, which is why the ordering belongs in the methods section.""")

    lit_tbl = lit.assign(f1=lit.f1.map(lambda v: f"{v:.4f}"))
    d.table(lit_tbl[["study", "venue", "dataset", "model", "f1", "protocol"]].rename(columns={
        "study": "Study", "venue": "Venue", "dataset": "Dataset", "model": "Best model",
        "f1": "Reported F1", "protocol": "Stated protocol"}),
        caption="""Reported minority-class F1 in recent employee-attrition studies, grouped by
        what each paper states about where resampling sits relative to the split. Values are as
        reported by the authors for the model they nominate as best. 'Not stated' means the
        paper does not describe the ordering, not that it is incorrect.""",
        fontsize=8, widths=[1.35, 1.15, .8, 1.2, .75, .95])

    # =====================================================================
    d.h1("5  Discussion")

    d.h2("5.1  Why a literature converges on numbers that look alike")
    d.p("""Sections of the applied literature on imbalanced classification report headline
        scores in a narrow high band across very different datasets, feature sets and model
        families. Read as measurements, that convergence is puzzling: problems that differ in
        difficulty should produce results that differ in quality. Section 4.2 supplies an
        explanation that requires no assumption of bad faith. Under the leaked ordering the
        reported number is not a measurement of the problem, and quantities that do not measure
        the problem do not vary with it. A band is exactly what one should expect.""")

    d.p("""The explanation is testable and we have tested part of it: the position of the band
        in our corpus depends on the classifier, and its width is a small fraction of the width
        of the honest distribution. It also makes a prediction that a reader can check against
        any literature they know well. Where a body of work reports scores that vary little
        across datasets of visibly different difficulty, the ordering of resampling and
        splitting is the first thing to look at.""")

    d.h2("5.2  Leakage as substitution, not inflation")
    d.p("""The most useful thing this study changes is a description. 'Over-optimistic' suggests
        a bias term: a true value plus an increment, so that a leaked result still carries
        information about the truth and could in principle be corrected. Our results say
        otherwise. The leaked score correlates with the honest score at only
        {corr_leaked_correct}, moves by {leaked_range_bydataset} across a corpus over which the
        honest score moves by {correct_range_bydataset}, and what movement it has runs in the
        wrong direction. There is little to correct, because what is there is a measurement of
        something else: of how much contamination the pipeline produced.""")

    d.p("""This also explains why the premium is so predictable. If the leaked score were the
        honest score plus noise, the difference would be noise. Because the leaked score is close
        to constant, the difference is essentially the honest score with its sign flipped - which
        is what the fit in Figure 2(c) shows, and why R-squared is {law_r2} rather than
        something modest.""")

    d.h2("5.3  What follows for review and editing")
    d.p("""Reviewers are not usually able to run the code, and often cannot see it. What they can
        do is ask a question whose answer is one sentence long, and notice when the answer is
        missing. Three concrete practices follow from our results.""")
    d.bullets([
        """Require the ordering to be stated. A methods section should say where resampling
        occurs relative to the split, and where it occurs relative to cross-validation folds.
        This is one sentence and it is currently absent from a large share of applied papers.""",
        """Treat a high score on a hard problem as a question, not a finding. Figure 5(b) makes
        the trade-off explicit: on a problem with an imbalance ratio near {ir_max}, a reported F1
        that would be unremarkable at low imbalance is already unusual.""",
        """Ask for the no-resampling baseline. It costs the authors one extra run, and in
        {pct_none_beats_correct}% of our matched comparisons it was the stronger of the two
        correctly-evaluated options. A paper that reports it has answered the question of
        whether resampling helped at all.""",
    ])

    d.h2("5.4  What follows for practice")
    d.p("""For practitioners the operative finding is Section 4.6. Full rebalancing is not
        required by anything, and it maximises the number of synthetic points in play. Targeting
        a minority prevalence of 0.25 rather than 1.0 cut the premium by
        {mitigation_reduction_pct}% in our corpus and reduced contamination in step. That
        is not a licence to resample before splitting; it is an argument that the default of
        full balance is a choice that costs something and should be justified.""")

    d.h2("5.5  Limitations")
    d.bullets([
        """Our corpus is assembled, not sampled. It spans a wide range of imbalance ratios by
        construction, and the distribution of problems within that range is not the distribution
        of problems in any field. Statements about the shape of the relationship are supported;
        statements about the typical magnitude in a given discipline are not.""",
        """Multi-class benchmarks are binarised one-versus-rest. That is the standard
        construction for imbalanced benchmark suites and it is how the sources we drew on were
        built, but it produces problems whose minority class is a coherent category, which may
        be easier than a genuinely heterogeneous minority.""",
        """Classifiers are untuned and datasets are capped at 5,000 rows for tractability. Both
        choices affect absolute scores. Neither should affect a difference between two pipelines
        that share the same model and the same data, which is what we report throughout.""",
        """The screen is trained on our corpus and inherits its coverage. Applied to a problem
        far outside the fitted ranges it extrapolates, and the released implementation says so
        in its output rather than returning a bare number.""",
        """We measure one leakage mechanism. Preprocessing fitted on the full dataset, feature
        selection before splitting, and duplicate records across the split are separate errors
        with separate signatures, and nothing here estimates their magnitude.""",
    ])

    d.h2("5.6  A reporting checklist")
    d.p("""The following five items would, if reported, let a reader evaluate any study that
        applies resampling to imbalanced data. None requires extra experiments.""")
    d.bullets([
        "Where the resampler sits relative to the split, and relative to cross-validation folds.",
        "The class balance of the test set as evaluated: real observations only, or resampled.",
        "The number of minority cases before resampling, not only the ratio.",
        "The score without resampling, as a baseline.",
        "The resampling target prevalence, if it is not full balance.",
    ], numbered=True)

    # =====================================================================
    d.h1("6  Conclusion")
    d.p("""Resampling before splitting inflates minority-class F1 by {mean_premium:.3f} on
        average across {n_datasets} datasets, seven classifier families and four over-samplers.
        That number is the least interesting result in this paper. The consequential one is that
        the inflation is not additive: under the leaked ordering the reported score stops
        varying with the difficulty of the problem, retaining {variance_ratio} times less
        variance than the honest score, and what variation remains runs against difficulty
        rather than with it. The premium is therefore predictable from difficulty alone, with
        R-squared {law_r2}, and the artefact leaves a signature stable enough to be recognised
        in results one did not produce, at AUC {diag_auc} under leave-one-dataset-out
        validation.""")

    d.p("""Two things follow that a practitioner can act on. Resampling to partial rather than
        full balance cuts the artefact by {mitigation_reduction_pct}% and gives a diagnostic
        handle on it, though it does not remove it. And a
        reported result can be screened from what papers already publish, which turns a general
        warning about leakage into a specific question about a specific study. We release the
        screen, the corpus builder and the full experimental record so that both can be checked
        and extended.""")

    # =====================================================================
    d.h1("Data and code availability")
    d.p("""All datasets used are public. The corpus builder, the experiment, the analysis and
        the manuscript build script are released together, along with the complete record of
        every model fit reported here and the machine-readable summary from which every number
        in this text was inserted. Repository and archival DOI to be inserted on acceptance.""")

    d.h1("Declarations")
    d.p("""Funding: none. Competing interests: the authors declare none. Ethics approval: not
        applicable; all data are public and de-identified. A companion study by the same authors
        examines this effect on a single dataset and is under review; it is cited here and a
        copy has been supplied to the editor.""")

    d.references(B.list())

    p = d.save(OUT)
    unresolved = [k for k in N.missing]
    if unresolved:
        raise SystemExit(f"unresolved placeholders: {sorted(unresolved)}")
    print(f"wrote {p}  ({p.stat().st_size/1024:.0f} kB), "
          f"{len(N.used)} summary keys used, {len(B.list())} references")
    return p


if __name__ == "__main__":
    build()
