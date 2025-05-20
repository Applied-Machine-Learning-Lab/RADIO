from flashrag.pipeline import SequentialPipeline

class CustomSequentialPipeline(SequentialPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        # parent class
        super().__init__(config, prompt_template, retriever, generator)

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question

        retrieval_results, scores = self.retriever.batch_search(input_query, return_score=True)
        dataset.update_output("retrieval_result", retrieval_results)
        dataset.update_output("retrieval_score", scores)

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if "llmlingua" in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                dataset.update_output("prompt", input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]

        else:
            input_prompts = [
                self.prompt_template.get_string(question=q, retrieval_result=r, answer=a)
                for q, r, a in zip(dataset.question, dataset.retrieval_result, dataset.golden_answers)
            ]
        dataset.update_output("prompt", input_prompts)

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc for doc in docs])
        # delete used refiner to release memory
        if self.refiner:
            del self.refiner
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset

class GenerateDatasetPipeline(SequentialPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template, retriever, generator)

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question

        retrieval_results, scores = self.retriever.batch_search(input_query, return_score=True)
        dataset.update_output("retrieval_result", retrieval_results)
        dataset.update_output("retrieval_score", scores)

        return dataset

class mmluPipeline(SequentialPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template, retriever, generator)

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question

        retrieval_results, scores = self.retriever.batch_search(input_query, return_score=True)
        dataset.update_output("retrieval_result", retrieval_results)
        dataset.update_output("retrieval_score", scores)

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if "llmlingua" in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                dataset.update_output("prompt", input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]

        else:
            # input_prompts = [
            #     self.prompt_template.get_string(question=q, retrieval_result=r, choices=choices)
            #     for q, r, choices in zip(dataset.question, dataset.retrieval_result, dataset.choices)
            # ]
            input_prompts = [
                self.prompt_template.get_string(question=q, retrieval_result=r)
                for q, r in zip(dataset.question, dataset.retrieval_result)
            ]
        dataset.update_output("prompt", input_prompts)

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc for doc in docs])
        # delete used refiner to release memory
        if self.refiner:
            del self.refiner
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset