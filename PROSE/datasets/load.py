from typing import Tuple

from PROSE.utils.helper_functions import get_base_dir_path


def get_data_filenames(dataset: str) -> Tuple[str, str, str, str]:
    base_dir = get_base_dir_path() / "PROSE/datasets/" / dataset
    if dataset == "bowlinggreen_41":
        data_filename = base_dir / "bowlinggreen_curated_postprocessed.csv"
        summaries_filename = (
            base_dir
            / "20241112-140604__summaries__v1__bowlinggreen_curated_postprocessed/logs.csv"
        )
        tags_filename = (
            base_dir
            / "20241112-140723_generate_tags_bowlinggreen_curated_postprocessed_50/logs.csv"
        )
        topic = "What do you think needs to be done to improve our hometown of Bowling Green, KY?"
    elif dataset == "drugs_80":
        data_filename = base_dir / "drugs_curated_no_brand_name_postprocessed_80.csv"
        summaries_filename = (
            base_dir
            / "20250116-213655__summaries__v1__drugs_curated_no_brand_name_postprocessed_80/logs.csv"
        )
        tags_filename = (
            base_dir
            / "20250116-214127_generate_tags_drugs_curated_no_brand_name_postprocessed_80_50/logs.csv"
        )
        topic = "What is your experience with the medication 'Ethinyl estradiol / norethindrone'?"
    elif dataset == "drugs_80_extreme_middle":
        data_filename = (
            base_dir / "drugs_curated_no_brand_name_postprocessed_extr_middle_80.csv"
        )
        summaries_filename = (
            base_dir
            / "20250128-135632__summaries__v1__drugs_curated_no_brand_name_postprocessed_extr_middle_80/logs.csv"
        )
        tags_filename = (
            base_dir
            / "20250128-141101_generate_tags_drugs_curated_no_brand_name_postprocessed_extr_middle_80_50/logs.csv"
        )
        topic = "What is your experience with the medication 'Ethinyl estradiol / norethindrone'?"
    elif dataset == "drugs_obesity_80":
        data_filename = (
            base_dir / "drugs_curated_no_brand_name_postprocessed_obesity_80.csv"
        )
        summaries_filename = (
            base_dir
            / "20250128-114923__summaries__v1__drugs_curated_no_brand_name_postprocessed_obesity_80/logs.csv"
        )
        tags_filename = (
            base_dir
            / "20250128-120003_generate_tags_drugs_curated_no_brand_name_postprocessed_obesity_80_50/logs.csv"
        )
        topic = "What is your experience with the medication 'Contrave'?"
    else:
        raise NotImplementedError

    return data_filename, summaries_filename, tags_filename, topic
