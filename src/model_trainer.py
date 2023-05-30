import pandas as pd
from aif360.sklearn.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR
from aif360.algorithms.inprocessing import GerryFairClassifier, MetaFairClassifier, PrejudiceRemover
from aif360.datasets import BinaryLabelDataset
from fairlearn.reductions import GridSearch, ExponentiatedGradient
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds, RejectOptionClassifierCV, PostProcessingMeta
from demv import DEMV


class ModelTrainer:

    def __init__(self, dataset: pd.DataFrame, label: str, sensitive_features: list, 
                  positive_label: int) -> None:
        self.dataset = dataset
        self.label = label
        self.sensitive_features = sensitive_features
        self.positive_label= positive_label

    def use_rw(self, model):
        data = self.dataset.set_index(self.sensitive_features)
        prot_attr = [s for s in self.sensitive_features]
        rw = Reweighing(prot_attr)
        _, weights = rw.fit_transform(
            data.drop(self.label, axis=1), data[self.label])
        model.fit(self.dataset.drop(columns=self.label), self.dataset[self.label], sample_weight=weights)
        return model

    def use_dir(self, model):
        bin_data = BinaryLabelDataset(favorable_label=self.positive_label,
                                      unfavorable_label=1-self.positive_label,
                                      df=self.dataset,
                                      label_names=[self.label],
                                      protected_attribute_names=self.sensitive_features)
        dir = DisparateImpactRemover(sensitive_attribute=self.sensitive_features[0])
        trans_data = dir.fit_transform(bin_data)
        new_data, _ = trans_data.convert_to_dataframe()
        model.fit(new_data.drop(columns=self.label), new_data[self.label])
        return model
    
    def use_demv(self, model):
      demv = DEMV(round_level=1)
      data = demv.fit_transform(
          self.dataset, self.sensitive_features, self.label)
      model.fit(data.drop(columns=self.label), data[self.label])
      return model



    def use_gerry(self):
      bin_data = BinaryLabelDataset(favorable_label=self.positive_label,
                                    unfavorable_label=1-self.positive_label,
                                    df=self.dataset,
                                    label_names=[self.label],
                                    protected_attribute_names=self.sensitive_features)
      gerry = GerryFairClassifier(fairness_def='FP')
      gerry.fit(bin_data)
      return gerry
    
    def use_meta(self):
      bin_data = BinaryLabelDataset(favorable_label=self.positive_label,
                                    unfavorable_label=1-self.positive_label,
                                    df=self.dataset,
                                    label_names=[self.label],
                                    protected_attribute_names=self.sensitive_features)
      meta = MetaFairClassifier(sensitive_attr=self.sensitive_features[0])
      meta.fit(bin_data)
      return meta

    def use_prj(self):
      bin_data = BinaryLabelDataset(favorable_label=self.positive_label,
                                    unfavorable_label=1-self.positive_label,
                                    df=self.dataset,
                                    label_names=[self.label],
                                    protected_attribute_names=self.sensitive_features)
      prej = PrejudiceRemover(sensitive_attr=self.sensitive_features[0], class_attr=self.label)
      prej.fit(bin_data)
      return prej
    
    def use_cal_eo(self, model):
      self.dataset = self.dataset.set_index(self.sensitive_features[0])
      cal = CalibratedEqualizedOdds(prot_attr=self.sensitive_features[0])
      meta = PostProcessingMeta(model, cal)
      meta.fit(self.dataset.drop(self.label, axis=1), self.dataset[self.label])
      return meta

    def use_rej_opt(self, model):
      self.dataset = self.dataset.set_index(self.sensitive_features[0])
      rej = RejectOptionClassifierCV(prot_attr=self.sensitive_features[0], scoring='statistical_parity')
      meta = PostProcessingMeta(model, rej)
      meta.fit(self.dataset.drop(self.label, axis=1), self.dataset[self.label])
      return meta
    