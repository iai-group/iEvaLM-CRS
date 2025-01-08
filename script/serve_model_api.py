"""Serve the model API."""

import argparse
import logging
import random
from typing import Any, Dict

from src.model.crb_crs.recommender.movie_recommender import MovieRecommender
from src.model.crs_model import CRSModel
from src.model_api import ModelAPI

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-12s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments.

    Returns:
        Command line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="serve_model_api.py",
        description="Serve model API.",
    )

    parser.add_argument(
        "--crs_model",
        type=str,
        choices=["kbrd", "barcor", "unicrs", "chatgpt", "crbcrs"],
    )

    parser.add_argument(
        "--kg_dataset", type=str, choices=["redial", "opendialkg"]
    )

    parser.add_argument("--corpus", type=str)
    parser.add_argument("--domain", type=str)

    # model_detailed
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--entity_hidden_size", type=int)
    parser.add_argument("--num_bases", type=int, default=8)
    parser.add_argument("--context_max_length", type=int)
    parser.add_argument("--entity_max_length", type=int)

    # model
    parser.add_argument("--rec_model", type=str)
    parser.add_argument("--conv_model", type=str)
    parser.add_argument("--mle_model", type=str)

    # conv
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--encoder_layers", type=int)
    parser.add_argument("--decoder_layers", type=int)
    parser.add_argument("--text_hidden_size", type=int)
    parser.add_argument("--attn_head", type=int)
    parser.add_argument("--resp_max_length", type=int)

    # prompt
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--text_tokenizer_path", type=str)
    parser.add_argument("--text_encoder", type=str)

    # server
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="5005")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def get_model_args(
    model_name: str, args: argparse.Namespace
) -> Dict[str, Any]:
    """Returns model's arguments from command line arguments.

    Args:
        model_name: Model's name.
        args: Command line arguments.

    Raises:
        ValueError: If model is not supported.

    Returns:
        Model's arguments.
    """
    if model_name == "kbrd":
        return {
            "debug": args.debug,
            "kg_dataset": args.kg_dataset,
            "hidden_size": args.hidden_size,
            "entity_hidden_size": args.entity_hidden_size,
            "num_bases": args.num_bases,
            "rec_model": args.rec_model,
            "conv_model": args.conv_model,
            "context_max_length": args.context_max_length,
            "entity_max_length": args.entity_max_length,
            "tokenizer_path": args.tokenizer_path,
            "encoder_layers": args.encoder_layers,
            "decoder_layers": args.decoder_layers,
            "text_hidden_size": args.text_hidden_size,
            "attn_head": args.attn_head,
            "resp_max_length": args.resp_max_length,
            "seed": args.seed,
        }
    elif model_name == "barcor":
        return {
            "debug": args.debug,
            "kg_dataset": args.kg_dataset,
            "rec_model": args.rec_model,
            "conv_model": args.conv_model,
            "context_max_length": args.context_max_length,
            "resp_max_length": args.resp_max_length,
            "tokenizer_path": args.tokenizer_path,
            "seed": args.seed,
        }
    elif model_name == "unicrs":
        return {
            "debug": args.debug,
            "seed": args.seed,
            "kg_dataset": args.kg_dataset,
            "tokenizer_path": args.tokenizer_path,
            "context_max_length": args.context_max_length,
            "entity_max_length": args.entity_max_length,
            "resp_max_length": args.resp_max_length,
            "text_tokenizer_path": args.text_tokenizer_path,
            "rec_model": args.rec_model,
            "conv_model": args.conv_model,
            "model": args.model,
            "num_bases": args.num_bases,
            "text_encoder": args.text_encoder,
        }
    elif model_name == "chatgpt":
        return {
            "seed": args.seed,
            "debug": args.debug,
            "kg_dataset": args.kg_dataset,
        }
    elif model_name == "crbcrs":
        return {
            "dataset": args.kg_dataset,
            "domain": args.domain,
            "corpus_folder": args.corpus,
            "mle_model_path": args.mle_model,
            "recommender_path": args.rec_model,
        }
    raise ValueError(f"Model {model_name} is not supported.")


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    model_args = get_model_args(args.crs_model, args)
    logger.info(f"Loaded arguments for {args.crs_model} model.")
    logger.debug(f"Model arguments:\n{model_args}")

    # Load model
    crs_model = CRSModel(crs_model=args.crs_model, **model_args)
    logger.info(f"Loaded {args.crs_model} model.")

    # Generation arguments
    response_generation_args = {}
    if args.crs_model == "unicrs":
        response_generation_args = {
            "movie_token": "<mask>",
        }

    # Start CRS API
    crs_server = ModelAPI(crs_model, args.kg_dataset, response_generation_args)
    crs_server.start(args.host, args.port)
