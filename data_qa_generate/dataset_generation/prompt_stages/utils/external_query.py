import abc
import requests


class ExternalQuery(abc.ABC):

    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def query(self, query: str) -> str:
        pass

    @abc.abstractmethod
    def query_with_context(self, query: str, **context) -> str:
        pass


class ExternalQueryAPI(ExternalQuery):
    def __init__(self, **kwargs):
        if "model" in kwargs:
            self.model = kwargs["model"]
        else:
            self.model = ""

    def query(self, query: str) -> str:
        res = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": query}],
            },
        )
        res_json = res.json()
        return res_json["choices"][0]["message"]["content"]

    def query_with_context(self, query: str, **context) -> str:
        if "img" in context:
            img_path = context["img"]

            # make it to base64
            import base64

            with open(img_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            img_path = f"data:image/jpeg;base64,{encoded_image}"
            res = requests.post(
                "http://localhost:8000/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": img_path}},
                                {"type": "text", "text": query},
                            ],
                        }
                    ],
                },
            )
            res_json = res.json()
            return res_json["choices"][0]["message"]["content"]
        else:
            return self.query(query)


class ExternalQueryOpenAI(ExternalQuery):
    def __init__(self, **kwargs):
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://api-inference.modelscope.cn/v1/",
            api_key="",  # ModelScope Token
        )

    def query(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",  # ModelScope Model-Id
            messages=[{"role": "user", "content": query}],
            stream=False,
        )
        return response.choices[0].message.content or ""

    def query_with_context(self, query: str, **context) -> str:
        if "img" in context:
            img_paths = context["img"]
            if isinstance(img_paths, str):
                img_paths = [img_paths]

            from copy import deepcopy

            img_paths = deepcopy(img_paths)

            # make it to base64
            for i, img_path in enumerate(img_paths):
                import base64

                with open(img_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                img_paths[i] = f"data:image/jpeg;base64,{encoded_image}"

            response = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-72B-Instruct",  # ModelScope Model-Id
                messages=[
                    {
                        "role": "user",
                        "content": [
                            *[
                                {"type": "image_url", "image_url": {"url": img_path}}
                                for img_path in img_paths
                            ],
                            {"type": "text", "text": query},
                        ],
                    }
                ],
                stream=False,
            )
            return response.choices[0].message.content or ""
        else:
            return self.query(query)


def construct_external_query(model_name: str, **kwargs) -> ExternalQuery:
    if model_name == "Qwen/Qwen2.5-VL-72B-Instruct":
        return ExternalQueryOpenAI(**kwargs)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    construct_external_query(model_name="Qwen/Qwen2.5-VL-72B-Instruct")
