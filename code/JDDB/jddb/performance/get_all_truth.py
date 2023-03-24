# just for ref, delete later

class Result:
    TRUE_DISRUPTION_LABEL = "isDisurption"
    TRUE_DOWNTTIME_LABEL = "DownTime"
    SHOT_NO_HEADER = "shot_list"
    TRUE_DISRUPTION_HEADER = "true_disruption"
    TRUE_DISRUPTION_TIME_HEADER = "true_disruption_time"

    def get_all_truth_from_metadb(metaDB):
        shots = self.get_all_shots()
        for shot in shots:
            true_disurption = metaDB.get_labels()[TRUE_DISRUPTION_LABEL]
            true_downtime = metaDB.get_labels()[TRUE_DOWNTIME_LABEL]
            shot_result = self.result.loc[self.result[SHOT_NO_HEADER] == shot]
            shot_result[TRUE_DISRUPTION_TIME_HEADER] = true_disurption
            shot_result[TRUE_DISRUPTION_TIME_HEADER] = true_downtime

    def add(self, shot_no: List[int], predicted_disruption: List[int], predicted_disruption_time: List[float]):
        """
            check lenth
            check repeated shoot,call check_repeated()
            use returned shot_list to add
        Args:
            shot_no: a list of shot number
            predicted_disruption:   a list of value 0 or 1, is disruptive
            predicted_disruption_time: a list of predicted_disruption_time, unit :s
        """
        if not (len(shot_no) == len(predicted_disruption) == len(predicted_disruption_time)):
            raise ValueError('The inputs do not share the same length.')

        for i in range(len(shot_no)):
            shot = shot_no[i]
            if predicted_disruption[i] == 0:
                predicted_disruption_time[i] = -1
            self.result.loc[self.result[SHOT_NO_HEADER] == shot, [SHOT_NO_HEADER, TRUE_DISRUPTION_HEADER, TRUE_DISRUPTION_TIME_HEADER]] = \
                [shot_no[i], predicted_disruption[i], predicted_disruption_time[i]]

# %%

# %%
