from app.process.import_.agent.state import create_default_state
from app.process.import_.agent.main_graph import import_app

# 创建一个state (task_id/local_file_path/is_md_read_enabled)
state = create_default_state(task_id="008",local_file_path="烫金机使用.md",is_md_read_enabled=True)

# 动态测试
import_app.invoke(state)

# 静态测试
import_app.get_graph().print_ascii()