from langchain.prompts import PromptTemplate


prompt = (
    PromptTemplate.from_template(
        """
    너는 요리사야 내가 가진 재료들을 갖고 만들 수 있는 요리를 {개수}추천하고
    그 요리의 레시피를 제시해줘. 내가 가진 재료는 아래와 같아.
    <재료>
    {재료}
    """
    )
)

print('########### start prompt #############')
print(prompt)
print('########### end prompt #############')

response = prompt.invoke({"개수": 6, "재료": "사과, 잼"})

print('########### start response #############')
print(response)
print('########### end response #############')
