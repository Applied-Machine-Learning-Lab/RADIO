from openai import OpenAI
import pandas as pd
import argparse
from pandarallel import pandarallel

def generate(row, args):
    completion = client.chat.completions.create(
        model = args.model,
        messages = [
            {"role": "system", "content": "You are a professional QA assistant. Given a question and the ground truth answer, you can output the rationale why the ground truth answer is corrrect."},
            {"role": "user", "content": "Question: " + str(row['question'])},
            {"role": "user", "content": "Answer: " + str(row['golden_answers'])},
            {"role": "user", "content": "Rationale: "}
        ],
        max_completion_tokens=1000,
        n=1,
        seed=42,
        temperature=0.2,
        top_p=1,
    )
    row['reason'] = completion.choices[0].message.content
    return row

def main(args):
    # Load the data
    if args.dataset == 'nq':
        data = pd.read_json('dataset/{}/train.jsonl'.format(args.dataset), lines=True)
        # sample 20000 rows
        data = data.loc[:20000]
    elif args.dataset == 'triviaqa':
        data = pd.read_json('dataset/{}/train.jsonl'.format(args.dataset), lines=True)
        # sample 20000 rows
        data = data.loc[:20000]
    elif args.dataset == 'mmlu':
        data = pd.read_json('dataset/{}/train.jsonl'.format(args.dataset), lines=True)
        # sample 20000 rows
        data = data.loc[:20000]

    # Generate the data
    pandarallel.initialize(progress_bar=True, nb_workers=64)
    data = data.parallel_apply(lambda row: generate(row, args), axis=1)
    data.to_json('dataset/{}/reason/train_with_reason.jsonl'.format(args.dataset), lines=True, orient='records')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--api_key', type=str, default='sk-xxxx')
    argparser.add_argument('--base_url', type=str, default='https://xxx.xxx.xxx')
    argparser.add_argument('--dataset', type=str, default='nq')
    argparser.add_argument('--model', type=str, default='gpt-4o-mini')

    args = argparser.parse_args()

    client = OpenAI(
        api_key = args.api_key,
        base_url = args.base_url,
        timeout = 30.0, # default is 10 minutes
        max_retries = 3, # default is 2
    )

    main(args)



