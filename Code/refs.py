"""
refs.py -- one bibliography, shared by all three manuscripts.

Every entry here was checked against the publisher's own record while the
manuscripts were being written.  Cite by key; `bibliography(keys)` returns the
numbered list in citation order and `cite(key)` the bracket number, so the
numbering can never drift out of step with the text.
"""
from __future__ import annotations

REFS: dict[str, str] = {
    # --- leakage --------------------------------------------------------
    "kaufman2012":
        "Kaufman, S., Rosset, S., Perlich, C., Stitelman, O. (2012). Leakage in data "
        "mining: formulation, detection, and avoidance. ACM Transactions on Knowledge "
        "Discovery from Data 6(4), 15:1-21.",
    "kapoor2023":
        "Kapoor, S., Narayanan, A. (2023). Leakage and the reproducibility crisis in "
        "machine-learning-based science. Patterns 4(9), 100804.",
    "whalen2022":
        "Whalen, S., Schreiber, J., Noble, W.S., Pollard, K.S. (2022). Navigating the "
        "pitfalls of applying machine learning in genomics. Nature Reviews Genetics 23, "
        "169-181.",
    "vandewiele2021":
        "Vandewiele, G., Dehaene, I., Kovacs, G., Sterckx, L., Janssens, O., Ongenae, F., "
        "De Backere, F., De Turck, F., Roelens, K., Decruyenaere, J., Van Hoecke, S., "
        "Demeester, T. (2021). Overly optimistic prediction results on imbalanced data: "
        "a case study of flaws and benefits when applying over-sampling. Artificial "
        "Intelligence in Medicine 111, 101987.",

    # --- resampling -----------------------------------------------------
    "chawla2002":
        "Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P. (2002). SMOTE: "
        "Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence "
        "Research 16, 321-357.",
    "han2005":
        "Han, H., Wang, W.-Y., Mao, B.-H. (2005). Borderline-SMOTE: a new over-sampling "
        "method in imbalanced data sets learning. In Advances in Intelligent Computing "
        "(ICIC 2005), LNCS 3644, 878-887.",
    "he2008":
        "He, H., Bai, Y., Garcia, E.A., Li, S. (2008). ADASYN: Adaptive synthetic "
        "sampling approach for imbalanced learning. In IEEE International Joint "
        "Conference on Neural Networks (IJCNN), 1322-1328.",
    "blagus2013":
        "Blagus, R., Lusa, L. (2013). SMOTE for high-dimensional class-imbalanced data. "
        "BMC Bioinformatics 14, 106.",
    "elor2022":
        "Elor, Y., Averbuch-Elor, H. (2022). To SMOTE, or not to SMOTE? "
        "arXiv:2201.08528.",
    "goorbergh2022":
        "van den Goorbergh, R., van Smeden, M., Timmerman, D., Van Calster, B. (2022). "
        "The harm of class imbalance corrections for risk prediction models: illustration "
        "and simulation using logistic regression. Journal of the American Medical "
        "Informatics Association 29(9), 1525-1534.",
    "carriero2025":
        "Carriero, A., Luijken, K., de Hond, A., Moons, K.G.M., van Calster, B., "
        "van Smeden, M. (2025). The harms of class imbalance corrections for machine "
        "learning based prediction models: a simulation study. Statistics in Medicine "
        "44(3-4), e10320.",

    # --- benchmarking and metrics ---------------------------------------
    "olson2017":
        "Olson, R.S., La Cava, W., Orzechowski, P., Urbanowicz, R.J., Moore, J.H. (2017). "
        "PMLB: a large benchmark suite for machine learning evaluation and comparison. "
        "BioData Mining 10, 36.",
    "romano2022":
        "Romano, J.D., Le, T.T., La Cava, W., Gregg, J.T., Goldberg, D.J., Chakraborty, P., "
        "Ray, N.L., Himmelstein, D., Fu, W., Moore, J.H. (2022). PMLB v1.0: an open-source "
        "dataset collection for benchmarking machine learning methods. Bioinformatics "
        "38(3), 878-880.",
    "demsar2006":
        "Demsar, J. (2006). Statistical comparisons of classifiers over multiple data "
        "sets. Journal of Machine Learning Research 7, 1-30.",
    "saito2015":
        "Saito, T., Rehmsmeier, M. (2015). The precision-recall plot is more informative "
        "than the ROC plot when evaluating binary classifiers on imbalanced datasets. "
        "PLoS ONE 10(3), e0118432.",
    "chicco2020":
        "Chicco, D., Jurman, G. (2020). The advantages of the Matthews correlation "
        "coefficient (MCC) over F1 score and accuracy in binary classification "
        "evaluation. BMC Genomics 21, 6.",

    # --- software -------------------------------------------------------
    "pedregosa2011":
        "Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: machine "
        "learning in Python. Journal of Machine Learning Research 12, 2825-2830.",
    "lemaitre2017":
        "Lemaitre, G., Nogueira, F., Aridas, C.K. (2017). Imbalanced-learn: a Python "
        "toolbox to tackle the curse of imbalanced datasets in machine learning. Journal "
        "of Machine Learning Research 18(17), 1-5.",
    "chen2016":
        "Chen, T., Guestrin, C. (2016). XGBoost: a scalable tree boosting system. In "
        "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge "
        "Discovery and Data Mining, 785-794.",

    # --- HR analytics: data ---------------------------------------------
    "alqahtani2025":
        "Alqahtani, H. (2025). Saudi Employee Attrition Dataset. Mendeley Data, V1. "
        "doi:10.17632/6z2hty8php.1",
    "ibmhr":
        "IBM Watson Analytics (2016). IBM HR Analytics Employee Attrition & Performance. "
        "Public dataset, distributed via Kaggle.",

    # --- HR analytics: reported results ---------------------------------
    "fallucchi2020":
        "Fallucchi, F., Coladangelo, M., Giuliano, R., De Luca, E.W. (2020). Predicting "
        "employee attrition using machine learning techniques. Computers 9(4), 86.",
    "cavescu2026":
        "Cavescu, A.M., Popescu, A.N. (2026). Leakage-free evaluation for employee "
        "attrition prediction on tabular data. Information 17(3), 308.",
    "alyousef2026":
        "Alyousef, M.I., Khan, H.W., Sattar, M.U. (2026). A hybrid predictive model for "
        "employee turnover: integrating ensemble learning and feature-driven insights "
        "from IBM HR Analytics. Information 17(2), 208.",
    "baydili2025":
        "Baydili, I.T., Tasci, B. (2025). Predicting employee attrition: XAI-powered "
        "models for managerial decision-making. Systems 13(7), 583.",
    "nassreddine2026":
        "Nassreddine, G., Hammoud, J., Al-Khatib, O., Al Majzoub, M. (2026). Employee "
        "attrition prediction: an explanatory and statistically robust ensemble learning "
        "model. Computers 15(3), 185.",
    "alali2026":
        "AL-Ali, M., Alwateer, M., Alsaedi, S.A., Balaha, H.M., Badawy, M., "
        "Elhosseini, M.A. (2026). Integrating machine learning and explainable AI for "
        "employee attrition prediction in HR analytics. Scientific Reports 16, 6344.",
    "hamja2025":
        "Hamja, A., Hasan, M., Hassan, Z., Uddin, M.P., Siddikee, M.J.A. (2025). An "
        "explainable machine learning-based employee attrition predictive system. Annals "
        "of Data Science. doi:10.1007/s40745-025-00617-9",

    # --- companion ------------------------------------------------------
    "paper1":
        "Companion study by the present authors: a single-dataset audit of the "
        "resampling-before-splitting artefact in employee attrition prediction. Under "
        "review; preprint available from the authors.",
}


class Bib:
    """Assigns reference numbers in order of first citation."""

    def __init__(self):
        self.order: list[str] = []

    def c(self, *keys: str) -> str:
        nums = []
        for k in keys:
            if k not in REFS:
                raise KeyError(f"unknown reference {k!r}")
            if k not in self.order:
                self.order.append(k)
            nums.append(str(self.order.index(k) + 1))
        return "[" + ", ".join(nums) + "]"

    def list(self) -> list[str]:
        return [REFS[k] for k in self.order]
