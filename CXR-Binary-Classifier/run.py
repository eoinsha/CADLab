import argparse
import logging

from aws_test_densenet import run_test

logging.basicConfig()

parser = argparse.ArgumentParser(description='Run AWS image test')
parser.add_argument('--key', required=True)
parser.add_argument('--label', required=True)

logger = logging.getLogger(__name__)

args = parser.parse_args()
logging.info(f'Running with args {args}')
jobs = [{'Key': args.key, 'Label': args.label}]
run_test(jobs)
