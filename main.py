import inquirer
import rich_click as click

from dotenv import load_dotenv, find_dotenv

from utils.items import MultiModels
from utils.operation import get_store_json_path


def ask_user_choice():
    choices = [option.value for option in MultiModels]

    question = [
        inquirer.List(
            "choice",
            message="Please choose a space agent",
            choices=choices,
        )
    ]
    answer = inquirer.prompt(question)

    selected_option = next(opt for opt in MultiModels if opt.value == answer["choice"])
    return selected_option


@click.command()
@click.option("-e", "--env", default=".env", type=click.Path(exists=True), help="Path to the .env file.")
@click.option("-d", "--db", default="beijing1", type=click.STRING, help="database name in neo4j")
@click.option("-s", "--step", default=35, type=click.INT, help="max steps for simulation in single epoch")
@click.option("-g", "--gt", default="example/ny-hk1-sh/final_ny_QA.json",
              type=click.Path(exists=True), help="Path to the .json file.")
@click.option("-r", "--repeat", default=1, type=click.INT, help="repeat nums for one question.")
def main(env: str, db: str, step: int, gt: str, repeat: int):
    agent = ask_user_choice()
    store_json = get_store_json_path(gt, db, agent, "INFO")
    from utils.map_logger import logger

    logger.info(f"Simulation trajectories will be saved into file {store_json}")

    if load_dotenv(find_dotenv(env, raise_error_if_not_found=True), verbose=True, override=True):
        logger.success(f"Loaded environment variables from {env}.")
    else:
        logger.error(f"Failed to load environment variables from {env}.")
    from src.map import Map

    street_map = Map.from_json(
        db_name=db,
        gt_json=gt,
        agent=agent,
        backtrack=True,
        backtrack_steps=3,
        backtrack_mechanism="confidence",
        use_backtrack_prompt=False,
        retrieve=False,
        retrieve_epoch=3,
        retrieve_method="topology",
        retrieve_distance=1,
        use_history_trajectory=False,
        history_steps=3
    )
    street_map.run(
        max_steps=step, repeat_num_for_single_question=repeat,
    )


if __name__ == '__main__':
    main()
