from pyrit.pyrit_tester import PyritTester
from pyrit.common import IN_MEMORY, initialize_pyrit, RDS
from pyrit.prompt_target import OpenAIChatTarget

import asyncio, sys, json

def init_pyrit():
    initialize_pyrit(memory_db_type=IN_MEMORY)
    #initialize_pyrit(memory_db_type=RDS, db_type="postgresql", schema_name="pyrit")
    assistant_target = OpenAIChatTarget()
    target = OpenAIChatTarget()

    tester = PyritTester(target, assistant_target)
    return tester

if __name__ == '__main__':
    tester = init_pyrit()
    print(tester.enum_tech())
    #asyncio.run(tester.tech_base64(tester.fetch_chinese_harm_dataset(), labels={"mytest": "refactor"}))