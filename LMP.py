
from global_param import *
import openai
from openai import RateLimitError, APIConnectionError
import re
import json
import cv2
import base64
import copy
from time import sleep
# imports for LMPs
import shapely
import ast
import astunparse
from time import sleep
from shapely.geometry import *
from shapely.affinity import *
from openai import RateLimitError, APIConnectionError
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from IPython.display import display, Image, Audio
from env_utils import LMP_wrapper


openai_api_key = 'sk-wjI23SNEp3giHqfy948979B05eA0499bBa68604959888403'
openai_base_url = 'https://api.v36.cm/v1/'
openai_default_headers = {"x-foo": "true"}

openai.api_key = openai_api_key
openai.base_url = openai_base_url
openai.default_headers = {"x-foo": "true"}


def get_video_description(video_address, user_query):
    video = cv2.VideoCapture(video_address)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    print(len(base64Frames), "frames read.")
    display_handle = display(None, display_id=True)
    for img in base64Frames:
        display_handle.update(Image(data=base64.b64decode(img.encode("utf-8"))))
        sleep(0.025)

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                user_query,
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]),
            ],
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 200,
    }
    result = openai.chat.completions.create(**params)
    return result


def extract_between(original_string, start_str, end_str):
    # 构建正则表达式
    pattern = re.escape(start_str) + r'(.*?)' + re.escape(end_str)

    # 使用 re.search 查找匹配的内容
    match = re.search(pattern, original_string, re.DOTALL)

    if match:
        return match.group(1)  # 返回两个特定字符串之间的内容
    return ''  # 如果没有找到，返回空字符串

def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except:
        exists = False
    else:
        exists = True
    return exists


def merge_dicts(dicts):
    return {
        k : v
        for d in dicts
        for k, v in d.items()
    }


def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['__']
    for phrase in banned_phrases:
        assert phrase not in code_str

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    exec(code_str, custom_gvars, lvars)

"""### LMP Utils"""

def setup_LMP(env, cfg_tabletop):
    # LMP env wrapper
    cfg_tabletop = copy.deepcopy(cfg_tabletop)
    cfg_tabletop['env'] = dict()
    cfg_tabletop['env']['init_objs'] = list(env.obj_name_to_id.keys())
    cfg_tabletop['env']['coords'] = lmp_tabletop_coords
    LMP_env = LMP_wrapper(env, cfg_tabletop)
    # creating APIs that the LMPs can interact with
    fixed_vars = {
        'np': np
    }
    fixed_vars.update({
        name: eval(name)
        for name in shapely.geometry.__all__ + shapely.affinity.__all__
    })
    variable_vars = {
        k: getattr(LMP_env, k)
        for k in [
            'get_bbox', 'get_obj_pos', 'get_color', 'is_obj_visible', 'denormalize_xy',
            'put_first_on_second', 'get_obj_names',
            'get_corner_name', 'get_side_name',
        ]
    }
    variable_vars['say'] = lambda msg: print(f'robot says: {msg}')

    # creating the function-generating LMP
    lmp_fgen = LMPFGen(cfg_tabletop['lmps']['fgen'], fixed_vars, variable_vars)

    # creating other low-level LMPs
    variable_vars.update({
        k: LMP(k, cfg_tabletop['lmps'][k], lmp_fgen, fixed_vars, variable_vars)
        for k in ['parse_obj_name', 'parse_position', 'parse_question', 'transform_shape_pts']
    })

    # creating the LMP that deals w/ high-level language commands
    lmp_tabletop_ui = LMP(
        'tabletop_ui', cfg_tabletop['lmps']['tabletop_ui'], lmp_fgen, fixed_vars, variable_vars
    )

    return lmp_tabletop_ui
class LMP:

    def __init__(self, name, cfg, lmp_fgen, fixed_vars, variable_vars):
        self._name = name
        self._cfg = cfg

        self._base_prompt = self._cfg['prompt_text']

        ##self._stop_tokens = list(self._cfg['stop'])

        self._lmp_fgen = lmp_fgen

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query, context=''):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session']:
            prompt += f'\n{self.exec_hist}'

        if context != '':
            prompt += f'\n{context}'

        use_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}{"You should only generate python code"}'


        messages=[
          {"role": "system", "content": prompt},
          {"role": "user", "content": use_query}
        ]
        prompt += f'\n{use_query}'

        return prompt, use_query, messages

    def __call__(self, query, context='', **kwargs):
        prompt, use_query, messages = self.build_prompt(query, context=context)

        while True:
            try:
                response = openai.chat.completions.create(
                    messages=messages,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['engine'],
                    max_tokens=self._cfg['max_tokens']
                )
                response_dict = response.to_dict()
                with open('log.txt', 'a') as f:
                  print(json.dumps(response_dict, indent=4),file = f)
                f.close()
                code_str_raw = response.choices[0].message.content.strip()
                code_str = extract_between(code_str_raw, "```python", "```")
                if code_str=='':
                  code_str = code_str_raw
                break
            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        if self._cfg['include_context'] and context != '':
            to_exec = f'{context}\n{code_str}'
            to_log = f'{context}\n{use_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{use_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
        print(f'LMP {self._name} exec:\n\n{to_log_pretty}\n')

        new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
        self._variable_vars.update(new_fs)

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        if not self._cfg['debug_mode']:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_exec}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            return lvars[self._cfg['return_val_name']]


class LMPFGen:

    def __init__(self, cfg, fixed_vars, variable_vars):
        self._cfg = cfg

       ## self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self._base_prompt = self._cfg['prompt_text']

    def create_f_from_sig(self, f_name, f_sig, other_vars=None, fix_bugs=False, return_src=False):
        print(f'Creating function: {f_sig}')

        use_query = f'{self._cfg["query_prefix"]}{f_sig}{self._cfg["query_suffix"]}{"You should only generate python code"}'
        prompt = f'{self._base_prompt}\n{use_query}'
        messages=[
          {"role": "system", "content": self._base_prompt},
          {"role": "user", "content": use_query}
        ]

        while True:
            try:
                response = openai.chat.completions.create(
                    messages=messages,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['engine'],
                    max_tokens=self._cfg['max_tokens']
                )
                response_dict = response.to_dict()
                with open('log.txt', 'a') as f:
                  print(json.dumps(response_dict, indent=4), file=f)
                f.close()
                f_src_raw = response.choices[0].message.content.strip()
                f_src = extract_between(f_src_raw, "```python", "```")
                if f_src=='':
                  f_src = f_src_raw
                break
            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        if fix_bugs:
            f_src = openai.Edit.create(
                model='code-davinci-edit-001',
                input='# ' + f_src,
                temperature=0,
                instruction='Fix the bug if there is one. Improve readability. Keep same inputs and outputs. Only small changes. No comments.',
            )['choices'][0]['text'].strip()

        if other_vars is None:
            other_vars = {}
        gvars = merge_dicts([self._fixed_vars, self._variable_vars, other_vars])
        lvars = {}

        exec_safe(f_src, gvars, lvars)

        f = lvars[f_name]

        to_print = highlight(f'{use_query}\n{f_src}', PythonLexer(), TerminalFormatter())
        print(f'LMP FGEN created:\n\n{to_print}\n')

        if return_src:
            return f, f_src
        return f

    def create_new_fs_from_code(self, code_str, other_vars=None, fix_bugs=False, return_src=False):
        fs, f_assigns = {}, {}
        f_parser = FunctionParser(fs, f_assigns)
        f_parser.visit(ast.parse(code_str))
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        if other_vars is None:
            other_vars = {}

        new_fs = {}
        srcs = {}
        for f_name, f_sig in fs.items():
            all_vars = merge_dicts([self._fixed_vars, self._variable_vars, new_fs, other_vars])
            if not var_exists(f_name, all_vars):
                f, f_src = self.create_f_from_sig(f_name, f_sig, new_fs, fix_bugs=fix_bugs, return_src=True)

                # recursively define child_fs in the function body if needed
                f_def_body = astunparse.unparse(ast.parse(f_src).body[0].body)
                child_fs, child_f_srcs = self.create_new_fs_from_code(
                    f_def_body, other_vars=all_vars, fix_bugs=fix_bugs, return_src=True
                )

                if len(child_fs) > 0:
                    new_fs.update(child_fs)
                    srcs.update(child_f_srcs)

                    # redefine parent f so newly created child_fs are in scope
                    gvars = merge_dicts([self._fixed_vars, self._variable_vars, new_fs, other_vars])
                    lvars = {}

                    exec_safe(f_src, gvars, lvars)

                    f = lvars[f_name]

                new_fs[f_name], srcs[f_name] = f, f_src

        if return_src:
            return new_fs, srcs
        return new_fs

class FunctionParser(ast.NodeTransformer):

    def __init__(self, fs, f_assigns):
      super().__init__()
      self._fs = fs
      self._f_assigns = f_assigns

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.value.func).strip()
            self._f_assigns[f_name] = assign_str
        return node




