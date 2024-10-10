import litserve as ls
from fastembed import TextEmbedding
from ingestion import ingest_pdfs
from prompt_template import qa_prompt_tmpl_str
from retriever import Retriever
from vllm import LLM
from vllm.sampling_params import SamplingParams


class DocumentChatAPI(ls.LitAPI):
    def setup(self, device):
        model_name = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"
        # model_name = "meta-llama/Llama-3.2-1B-Instruct"
        # model_name = "Qwen/Qwen2.5-14B-Instruct-AWQ"
        self.llm = LLM(model=model_name, max_model_len=8000)
        # embed_model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        # ingest_pdfs("./data", embed_model)
        # self.retriever = Retriever(embed_model)

    def decode_request(self, request):
        return request["query"]

    def predict(self, query):
        # context = self.retriever.generate_context(query)
        # prompt = qa_prompt_tmpl_str.format(context=context, query=query)

        messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]

        sampling_params = SamplingParams(max_tokens=8192, temperature=0.7)
        outputs = self.llm.chat(messages=messages, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    def encode_response(self, output):
        return {"output": output}


if __name__ == "__main__":
    api = DocumentChatAPI()
    server = ls.LitServer(api)
    server.run(port=8080)
