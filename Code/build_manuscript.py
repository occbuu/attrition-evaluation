#!/usr/bin/env python3
"""
build_manuscript.py -- HR-facing paper on evaluation protocol for attrition models.

Target venue: a human-resource-management or information-systems journal with a
quantitative readership.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import literature
from paperkit import Numbers, Paper, table_csv, fmt, ROOT
from refs import Bib

OUT = ROOT / "output" / "manuscript.docx"


def build() -> Path:
    N = Numbers()
    B = Bib()
    d = Paper(N)

    d.title("What an attrition model is worth: evaluation protocol, not algorithm choice, "
            "determines reported performance in HR analytics")
    d.authors("Author names withheld for review", "Affiliations withheld for review")

    d.abstract(
        """Employee attrition prediction has become a standard application of machine learning
        in human resource analytics, and published accuracies are high: minority-class F1 scores
        above 0.9 are routine. We show that on the datasets these studies use, such scores are
        not attainable under a correctly specified evaluation, and that the gap is explained by
        a single, widespread and easily corrected protocol error - applying synthetic
        over-sampling to the whole dataset before splitting it into training and test sets. On
        {hr_n_datasets} public employee-turnover datasets, seven classifier families, four
        over-sampling methods and ten random seeds, we fit each model three ways while holding
        everything else fixed. Resampling before the split raises minority-class F1 by
        {hr_premium:.3f} on average (95% CI {hr_premium_ci_lo:.3f} to {hr_premium_ci_hi:.3f}),
        lifting it from an honest {hr_correct_mean} to {hr_leaked_mean}. How large the inflation
        is depends almost entirely on how imbalanced the workforce is (r = {hr_corr_ir_premium}
        across datasets): on a near-balanced dataset it is {hr_premium_min:.3f} and effectively
        harmless, while on the IBM benchmark, whose imbalance is typical of real attrition data,
        it is {hr_premium_max:.3f} - it multiplies the reported F1 by {ibm_ratio}. That is the
        practical finding: the error is largest exactly where attrition data actually sits.
        Under the correct protocol, mean F1 across these datasets ranges from {hr_correct_min}
        to {hr_correct_max} - a useful triage tool, not a solved problem. We set out what an HR
        analytics team should require of an attrition model, what a
        realistic performance expectation looks like, and why a headline F1 near 0.9 on an
        attrition dataset should prompt a question about the pipeline rather than a purchase
        decision.""")

    d.keywords("HR analytics; employee attrition; turnover prediction; class imbalance; "
               "SMOTE; model evaluation; data leakage")

    # =====================================================================
    d.h1("1  Introduction")
    d.p(f"""Predicting which employees will leave is one of the few HR problems with a clean
        supervised-learning formulation, and the literature has grown quickly around a small
        number of public datasets {B.c('fallucchi2020')}. The reported results are impressive.
        Studies on the IBM HR benchmark, a dataset of 1,470 employees of whom 237 left, have
        recently reported minority-class F1 scores of {float(literature.frame().f1.max()):.2f}
        and above {B.c('hamja2025', 'nassreddine2026')}.""")

    d.p("""A practitioner reading that literature would reasonably conclude that attrition is
        largely a solved prediction problem, and that the remaining work is deployment. This
        paper argues that the conclusion does not follow, and that the reason has nothing to do
        with which algorithm anyone chose.""")

    d.p("""Attrition datasets are imbalanced: leavers are the minority, typically between a tenth
        and a fifth of the workforce. The standard response is synthetic over-sampling, usually
        SMOTE, which manufactures additional minority cases by interpolating between real ones.
        Applied correctly, the over-sampler sees only the training data. Applied to the whole
        dataset before the split, it manufactures near-copies of employees who then end up on
        both sides of the split - so the model is asked, at test time, about people it has
        effectively already met.""")

    d.p("""The error is old news as a matter of principle. What has not been available to this
        field is a measurement of how much it changes the answer on the exact datasets the field
        uses, and therefore of what an honest performance expectation looks like. That is what
        this paper supplies.""")

    d.p(f"""We fit seven classifier families on {{hr_n_datasets}} public employee-turnover
        datasets, with four over-samplers, ten seeds, and three evaluation protocols that differ
        only in where the resampler sits. The datasets span workforces from
        {{hr_ir_min}}-to-1 to {{hr_ir_max}}-to-1 imbalance and include the IBM benchmark, a
        second corporate case study, a healthcare workforce, a technology-sector job-change
        dataset of nearly twenty thousand records, a Russian turnover dataset and survey data
        from 1,191 private-sector employees in Saudi Arabia {B.c('alqahtani2025')}. That last
        one matters: it is primary survey data rather than a re-publication of a synthetic
        corporate extract, and it is the closest thing the field has to a check that findings on
        the IBM benchmark are not artefacts of that benchmark's construction.""")

    d.p("""Three findings follow, and each has a direct managerial reading.""")
    d.bullets([
        """Reported performance in this literature is dominated by evaluation protocol, not by
        algorithm choice. Moving the resampler across the split changes minority-class F1 by
        {hr_premium:.3f}; the spread between the best and worst of the seven model families,
        correctly evaluated, is {hr_clf_spread_correct}.""",
        """Under a correct protocol, attrition prediction on these datasets is a moderate
        problem, not a solved one. Honest F1 ranges from {hr_correct_min} to {hr_correct_max}
        depending on the dataset.""",
        """The damage is concentrated where real attrition data sits. The premium tracks the
        imbalance ratio at r = {hr_corr_ir_premium}: near-balanced datasets are barely affected,
        and the datasets with realistic leaver proportions are affected most.""",
    ])

    # =====================================================================
    d.h1("2  Background")

    d.h2("2.1  Why attrition data is imbalanced, and what is usually done about it")
    d.p(f"""Annual voluntary turnover in most organisations runs between five and twenty per
        cent, so any dataset labelled at the employee-year level has a minority class of that
        size. Classifiers trained on such data tend to predict the majority class, and the
        standard remedy in the applied literature is to rebalance the training data with SMOTE
        {B.c('chawla2002')} or one of its variants {B.c('han2005', 'he2008')}.""")

    d.p(f"""Whether rebalancing helps at all is contested well outside HR analytics.
        Blagus and Lusa {B.c('blagus2013')} showed it is largely ineffective in high
        dimensions. Van den Goorbergh and colleagues {B.c('goorbergh2022')} showed that
        imbalance corrections destroy the calibration of
        clinical risk models - they make predicted probabilities wrong - while leaving
        discrimination unchanged, and Carriero and colleagues {B.c('carriero2025')} confirmed this for machine
        learning models by simulation. Elor and Averbuch-Elor {B.c('elor2022')} found no
        benefit from balancing for
        strong, properly tuned classifiers, though weaker learners still gain from it. Calibration is not a technicality for HR
        use: a retention budget allocated from a model that says an employee has a 70 per cent
        chance of leaving, when the true figure is 20 per cent, is misallocated in proportion to
        the error.""")

    d.h2("2.2  The protocol error")
    d.p(f"""Separately from whether to resample, there is the question of when. SMOTE creates a
        synthetic minority case by interpolating between a real minority case and one of its
        neighbours. Do this before splitting, and a synthetic employee that is nine-tenths a
        copy of employee A can end up in training while employee A ends up in the test set. The
        model does not need to generalise to A. It needs to remember the copy. The error is
        catalogued in the general leakage literature {B.c('kaufman2012', 'kapoor2023')} and was
        shown to inflate published results in perinatal medicine {B.c('vandewiele2021')}.""")

    d.p(f"""Within HR analytics the issue has begun to surface. Cavescu and Popescu
        {B.c('cavescu2026')} recently proposed a leakage-free evaluation protocol for attrition prediction, applying SMOTE
        strictly within cross-validation folds on the IBM benchmark, and reported a
        cross-validated F1 of 0.439 for logistic regression - a figure far below the field's
        headline numbers. Alyousef and colleagues {B.c('alyousef2026')}, also on the IBM benchmark and also
        stating that
        resampling was confined to the training data, report 0.5042. Neither study measures how
        large the difference is, because neither runs both protocols.""")

    # =====================================================================
    d.h1("3  Data and method")

    d.h2("3.1  Datasets")
    d.p("""We use {hr_n_datasets} public employee-turnover datasets, listed in Table 1. Each is
        harmonised the same way: identifier and constant columns dropped, nominal fields one-hot
        encoded, numeric fields median-imputed, and the minority class coded as positive. No
        feature selection is performed, because feature selection carried out on the full dataset
        is itself a leakage mechanism and we did not want to confound one with another.""")

    hr = table_csv("table8_hr.csv")
    d.table(
        fmt(hr, keep=["dataset", "n", "n_features", "n_minority", "imbalance_ratio"],
            dec={"imbalance_ratio": 2},
            rename={"dataset": "Dataset", "n": "Employees", "n_features": "Features",
                    "n_minority": "Leavers", "imbalance_ratio": "Imbalance ratio"}),
        caption="""The employee-turnover datasets. Feature counts are after one-hot encoding.
        The job-change dataset was stratified-subsampled to 5,000 records to keep the factorial
        tractable; class balance is unchanged.""",
        fontsize=9, widths=[1.6, 1.0, .9, .9, 1.2])

    d.h2("3.2  Three protocols, one difference")
    d.p("""Every comparison holds the dataset, the encoding, the classifier, its
        hyperparameters, the over-sampler, the split proportion and both random seeds fixed. The
        only thing that changes is where the over-sampler sits:""")
    d.bullets([
        "leaked - over-sample the whole dataset, then split 70:30;",
        "correct - split 70:30, then over-sample the training partition only;",
        "none - split 70:30 and do not over-sample.",
    ])
    d.p("""Because nothing else differs, any difference in the reported score is attributable to
        that position alone. This is worth emphasising for a management readership: the
        comparison here is not between methods, and no method is being criticised. It is between
        two ways of measuring the same method.""")

    d.h2("3.3  Models, metrics and scale")
    d.p("""Seven classifier families are used - logistic regression, a decision tree, 5-nearest
        neighbours, an RBF support vector machine, a random forest, gradient boosting and
        XGBoost - with library defaults, and four over-samplers: SMOTE, Borderline-SMOTE, ADASYN
        and random over-sampling. Ten random seeds allow every figure to carry a confidence
        interval. We report minority-class F1 because it is what this literature reports, and
        check the conclusions against Matthews correlation and average precision, which behave
        better under imbalance.""")

    d.p("""These six datasets sit in a corpus of {n_datasets} public binary classification
        problems; the remaining problems are used only as a reference distribution so that
        attrition can be compared with other domains. The present paper is confined to the HR
        datasets and to what an HR analytics function should do about the protocol.""")

    # =====================================================================
    d.h1("4  Results")

    d.h2("4.1  How much the protocol is worth")
    d.p("""Across the HR datasets, over-sampling before the split raises minority-class F1 by
        {hr_premium:.3f} on average, with a 95% confidence interval of {hr_premium_ci_lo:.3f}
        to {hr_premium_ci_hi:.3f}. Mean F1 goes from {hr_correct_mean} under the correct
        protocol to {hr_leaked_mean} under the leaked one.""")

    d.p("""To put that in the terms this literature uses: the protocol is worth
        {hr_premium:.3f} of F1, while the gap between the best and worst of the seven algorithm
        families - correctly evaluated, averaged over datasets, over-samplers and seeds - is
        {hr_clf_spread_correct}. A study that changes its evaluation protocol moves its headline
        number nearly twice as far as a study that changes its algorithm. Since papers compete
        on the headline number, this is a problem the field has an incentive not to notice.""")

    d.table(
        fmt(hr, keep=["dataset", "imbalance_ratio", "none", "correct", "leaked", "premium",
                      "premium_lo", "premium_hi"],
            dec={"imbalance_ratio": 2, "none": 4, "correct": 4, "leaked": 4,
                 "premium": 4, "premium_lo": 4, "premium_hi": 4},
            rename={"dataset": "Dataset", "imbalance_ratio": "IR", "none": "No resampling",
                    "correct": "Correct", "leaked": "Leaked", "premium": "Premium",
                    "premium_lo": "CI low", "premium_hi": "CI high"}),
        caption="""Minority-class F1 by dataset and protocol, averaged over seven classifiers,
        four over-samplers and ten seeds. 'Premium' is leaked minus correct, with a bootstrap
        95% interval. The premium rises steeply with the imbalance ratio, which is the point of
        Section 4.2.""",
        fontsize=8.5)

    d.figure("fig06_hr.png",
             """The three protocols on each employee-turnover dataset, averaged over all
             classifiers, over-samplers and seeds. Datasets are ordered by imbalance ratio.
             On the two near-balanced datasets at the left the three protocols agree; as the
             leaver proportion falls, the leaked bar detaches from the other two and the gap
             widens to a factor of {ibm_ratio} on the IBM benchmark.""", width=5.8)

    d.h2("4.2  The damage is concentrated where real attrition data sits")
    d.p("""Table 2 and Figure 1 show that the premium is not spread evenly. It runs from
        {hr_premium_min:.4f} on {hr_premium_min_dataset}, which is close to balanced, to
        {hr_premium_max:.4f} on {hr_premium_max_dataset}. Across the six datasets the premium
        tracks the imbalance ratio at r = {hr_corr_ir_premium}: how much a protocol error is
        worth is almost fully determined by how rare leavers are in the data.""")

    d.p("""The mechanism explains it. The more imbalanced the workforce, the more synthetic
        employees an over-sampler has to manufacture to balance the data, and so the more of the
        test set is present in the training partition as an interpolation. On a dataset where
        leavers are already half the sample, the over-sampler barely does anything and the error
        barely matters. On a dataset with a realistic attrition rate, it does a great deal.""")

    d.p("""Figure 2 puts the six datasets in the context of the wider benchmark corpus, and
        the placement inverts what looks like reassuring news. The employee-turnover datasets sit at
        the mild end of the imbalance range compared with general machine-learning benchmarks,
        and one might expect the error to matter less here. The opposite holds within the domain:
        the datasets that most resemble a real workforce are the ones where the protocol matters
        most. On the IBM benchmark the leaked protocol returns {ibm_ratio} times the honest
        score - not an increment, a multiple.""")

    d.figure("fig07_hr_vs_benchmark.png",
             """The HR datasets against the wider benchmark corpus. Each point is one matched
             comparison and the horizontal axis is the imbalance ratio. Within the HR range the
             premium climbs steeply with imbalance, which is why the datasets that most resemble
             a real workforce are the ones the protocol error damages most.""", width=4.6)

    d.h2("4.3  Whether to resample at all")
    d.p("""The 'no resampling' column of Table 2 answers a question the literature usually skips,
        and on these datasets the answer favours resampling. Correctly-ordered resampling beat
        the no-resampling baseline on all {hr_n_datasets} datasets, by {hr_none_minus_correct}
        on average, and the baseline won only {hr_pct_none_beats_correct}% of matched
        comparisons.""")

    d.p("""That is a narrower result than the wider benchmark literature would predict, and it is
        worth being precise about why. Across the full corpus of {n_datasets} datasets assembled
        here, no resampling beat correctly-ordered resampling in
        {pct_none_beats_correct}% of matched comparisons; the sceptical case against
        rebalancing is strongest at high imbalance and in high dimensions, and the HR datasets
        are at neither extreme. On employee-turnover data of this kind, resampling the training
        partition is a modest but real improvement.""")

    d.p("""The recommendation therefore is not to stop resampling. It is to report the baseline.
        It costs one extra run, it is what makes the rebalancing step defensible, and on other
        workforces - a large organisation with a two per cent attrition rate, say - it may well
        be the winner.""")

    d.h2("4.4  What honest performance looks like, dataset by dataset")
    ibm = "hr_by_dataset.ibm_hr"
    d.p(f"""On the IBM benchmark, the dataset most of this literature uses, our correctly
        evaluated mean F1 is {{{ibm}.correct}} and the no-resampling baseline is
        {{{ibm}.none}}; the leaked protocol returns {{{ibm}.leaked}}. Independent studies that
        state they confined resampling to the training data report 0.439
        {B.c('cavescu2026')} and 0.5042 {B.c('alyousef2026')} on the same data with a single
        tuned model, which brackets our untuned average across seven families. Studies that do
        not state their ordering report between
        {literature.frame().query("protocol=='not stated'").f1.min():.3f} and
        {literature.frame().query("protocol=='not stated'").f1.max():.3f}
        {B.c('alali2026', 'nassreddine2026', 'hamja2025')}.""")

    d.p("""Table 3 collects these figures. We are not in a position to say that any individual
        study is wrong, and we do not say
        it. Several innocent explanations exist for a high reported number: a macro-averaged F1
        reported without qualification, different preprocessing, different splits. But a reader
        cannot distinguish between those explanations and a protocol error, because the
        information needed to distinguish them is usually not in the paper. That is the finding
        an HR analytics reader should take away: the published figures cannot currently be used
        to set an expectation, and the fix is a single sentence in a methods section.""")

    lit = literature.frame()
    d.table(lit.assign(f1=lit.f1.map(lambda v: f"{v:.4f}"))[
        ["study", "venue", "dataset", "model", "f1", "protocol"]].rename(columns={
            "study": "Study", "venue": "Venue", "dataset": "Dataset", "model": "Best model",
            "f1": "Reported F1", "protocol": "Stated protocol"}),
        caption="""Reported minority-class F1 in recent employee-attrition studies, grouped by
        what each paper states about where resampling sits relative to the split. Values are as
        reported by the authors for the model they nominate as best. 'Not stated' means the
        paper does not describe the ordering; it is not a claim that the study is incorrect.""",
        fontsize=8, widths=[1.35, 1.15, .8, 1.2, .75, .95])

    # =====================================================================
    d.h1("5  What this means for HR analytics practice")

    d.h2("5.1  For teams commissioning or buying a model")
    d.p("""The five questions below can be asked of any vendor or internal team, require no
        technical background to ask, and cannot be answered evasively without that being
        obvious.""")
    d.bullets([
        """Was the test set built before or after any rebalancing? The answer should be
        'before'. If the answer is unclear, the reported number cannot be interpreted.""",
        """Does the test set contain any synthetic employees? It should not. A test set is a
        sample of people, not of interpolations.""",
        """What does the model score without rebalancing? If nobody has run it, the rebalancing
        step has not been justified.""",
        """How many leavers are in the data, not what percentage? A 5 per cent minority of
        20,000 employees and of 300 employees are different problems, and the second one usually
        cannot support a reliable model.""",
        """Are the predicted probabilities calibrated, and was calibration checked after
        rebalancing? Rebalancing systematically inflates predicted probabilities, which
        matters whenever the output drives a budget.""",
    ], numbered=True)

    d.h2("5.2  A realistic expectation")
    d.p("""Here is what the honest numbers mean in operational terms. Averaged over the six
        datasets and all seven model families, a correctly evaluated model recalls
        {hr_correct_recall} of leavers at a precision of {hr_correct_precision}: roughly three
        in five of the people it flags do leave, and it finds roughly three in five of those who
        do. Per-dataset recall runs from {hr_correct_recall_min} to {hr_correct_recall_max}.
        Ranking quality is better than the thresholded numbers suggest - ROC-AUC averages
        {hr_correct_rocauc} and average precision {hr_correct_prauc} - which is the useful
        property, because a retention programme works from a ranked list rather than from a
        yes/no label.""")

    d.p("""For comparison, the leaked protocol reports precision {hr_leaked_precision} and
        recall {hr_leaked_recall} on the same data with the same models. An HR director shown
        those figures would reasonably plan for a system that catches five leavers in six and is
        right four times in five. The deployed system will catch three in five and be right
        three times in five. The distance between those two operating pictures is the practical
        cost of the error.""")

    d.p("""This is not a counsel of despair. A model with an ROC-AUC near {hr_correct_rocauc}
        concentrates risk substantially relative to a random sample of the workforce, and the
        business case for retention programmes rarely depends on individual-level certainty. The
        problem with the inflated numbers is not that they are optimistic but that they set an
        expectation no deployed system can meet, which is how analytics functions lose
        credibility after a launch.""")

    d.h2("5.3  For authors and reviewers in this field")
    d.p("""Report where the resampler sits. Report the number of leavers, not only the
        percentage. Report the no-resampling baseline. Report calibration. None of these
        requires additional data collection, and together they would make the published
        literature interpretable in a way it currently is not.""")

    d.h2("5.4  Limitations")
    d.bullets([
        """Six datasets is a small sample of workforces, all public, none of them a large
        contemporary enterprise HRIS extract. The imbalance ratios run from {hr_ir_min} to
        {hr_ir_max}, which is representative of published attrition studies but not of every
        organisation.""",
        """Models are untuned, so absolute scores are below what a tuned pipeline would reach.
        Every number we report is a difference between two evaluations of the same untuned
        model, which is unaffected by that choice, but readers should not read the honest column
        of Table 2 as an upper bound on achievable performance.""",
        """We measure one protocol error. Feature selection performed on the full dataset,
        preprocessing fitted before the split, and duplicate employee-years across the split are
        separate problems with separate magnitudes.""",
        """We report discrimination, not calibration or business value. The calibration
        literature cited above suggests rebalancing harms the first; quantifying the second
        requires cost data no public attrition dataset carries.""",
    ])

    # =====================================================================
    d.h1("6  Conclusion")
    d.p("""On {hr_n_datasets} public employee-turnover datasets, applying synthetic
        over-sampling before rather than after the train-test split raises reported
        minority-class F1 by {hr_premium:.3f}, moving it from an honest {hr_correct_mean} to
        {hr_leaked_mean}. The inflation is larger than the spread between the seven algorithm
        families we tested, which means that in this literature the choice of evaluation
        protocol matters more to reported performance than the choice of model. It is also
        concentrated where it does the most damage: the premium tracks the imbalance ratio at
        r = {hr_corr_ir_premium}, so the datasets that most resemble a real workforce are the
        ones the error distorts most. On the IBM benchmark it multiplies the reported score by
        {ibm_ratio}.""")

    d.p("""A realistic expectation for attrition prediction on data of this kind is a
        moderately discriminating triage tool, not an individual-level classifier. Getting to
        that expectation requires no new methods - only that the ordering of resampling and
        splitting be stated, and that the no-resampling baseline be reported alongside
        it.""")

    d.h1("Data and code availability")
    d.p("""All six datasets are public. The harmonisation code, the full experimental record and
        the machine-readable summary from which every number in this paper was inserted are
        in this repository. Archival DOI to be inserted on acceptance.""")

    d.h1("Declarations")
    d.p("""Funding: none. Competing interests: the authors declare none. Ethics approval: not
        applicable; all datasets are public and de-identified.""")

    d.references(B.list())
    p = d.save(OUT)
    if N.missing:
        raise SystemExit(f"unresolved placeholders: {sorted(N.missing)}")
    print(f"wrote {p}")
    return p


if __name__ == "__main__":
    build()
