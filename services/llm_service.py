from utils.prompt_loader import load_prompt
from errors.exceptions import LLMServiceError
from utils.logger import logger

class LLMService:

    def __init__(self, groq_client):
        self.client = groq_client

    def answer_question(self, context: str, question: str) -> str:
        try:
            template = load_prompt("qa.txt")
            prompt = template.format(context=context, question=question)

            return self.client.complete(
                prompt=prompt,
                max_tokens=300,
                temperature=0.1
            )
        except Exception as e:
            logger.exception("QA prompt failed")
            raise LLMServiceError(str(e))

    def summarize(self, text: str) -> str:
        try:
            template = load_prompt("summary.txt")
            prompt = template.format(text=text)

            return self.client.complete(
                prompt=prompt,
                max_tokens=400,
                temperature=0.2
            )
        except Exception as e:
            logger.exception("Summary prompt failed")
            raise LLMServiceError(str(e))

    def generate_mindmap(self, text: str) -> str:
        try:
            template = load_prompt("mindmap.txt")
            prompt = template.format(text=text)

            return self.client.complete(
                prompt=prompt,
                max_tokens=800,
                temperature=0.2
            )
        except Exception as e:
            logger.exception("Mindmap prompt failed")
            raise LLMServiceError(str(e))
