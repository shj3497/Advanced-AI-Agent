from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

res1 = RunnablePassthrough.assign(
    mult=lambda x: x["num"]*3).invoke({"num": 30})
print('res1 : ', res1)

runnable = RunnableParallel(
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"]*3),
    modified=lambda x: x["num"] + 1,
)

res2 = runnable.invoke({"num": 10})

print('res2 : ', res2)


def add_smile(x):
    return x + ":)"


# 함수를 RunnableLambda 안에 넣어주면 Runnable 객체로 사용할 수 있다?
add_smile = RunnableLambda(add_smile)
