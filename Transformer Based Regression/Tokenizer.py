import torch

class RLLMTokenizer:

  def __init__(self,space):
    self.exp_tokens = self.list_exponents(space)
    self.vocab_size = len(self.exp_tokens)+12

  def list_exponents(self,s):

    pos_tokens = [f"E{x}" for x in range(s)]
    neg_tokens = [f"E-{x}" for x in range(1,s)]
    exponents = pos_tokens + neg_tokens
    exp_tokens = {}
    exp_tokens['+'] = 10
    exp_tokens['-'] = 11
    it = 0
    for i in exponents:
      exp_tokens[i] = 12 + it
      it += 1
    exp_tokens['[START]']=max(exp_tokens.values())+1
    return exp_tokens

  def get_exp_id(self,exp):
    key = f"E{exp:d}"   # e.g. E+3, E-2
    if key not in self.exp_tokens:
        return None
    return self.exp_tokens[key]

  def get_exp_num(self,exp_id):
    for k, v in self.exp_tokens.items():
        if v == exp_id:
          return k
    return None

  def encode(self, num_str, num_digits=3):
    if not isinstance(num_str, str):
      num_str = f"{num_str}"
    # remove sign (we’ll handle later)
    sign = "+"
    if num_str.startswith("-"):
      sign = "-"
      num_str = num_str[1:]

    # split
    if "." in num_str:
      int_part, frac_part = num_str.split(".")
    else:
      int_part, frac_part = num_str, ""

    frac_part += "0"*num_digits # complete the missing digits with zeros
    digits = int_part + frac_part

    # ----- compute mantissa (num_digits default value 3 digits) -----
    mantissa = digits[:num_digits]              # exactly 3 digits
    mantissa_tokens = [int(i) for i in mantissa]

    # ----- compute exponent -----
    exp = (len(digits) - num_digits) - len(frac_part)
    exp_id = self.get_exp_id(exp)
    # ----- add sign -----
    mantissa_tokens.insert(0, self.exp_tokens[sign])
    mantissa_tokens.insert(0, self.exp_tokens['[START]'])

    # ----- final tokens -----
    return torch.tensor(mantissa_tokens + [exp_id],dtype=torch.int64)

  def decode(self, tokens=None):
    if tokens is None:
      return None
    if not isinstance(tokens, list):
      tokens = tokens.tolist()
    try:
      tokens[1] = self.get_exp_num(tokens[1])
      number  = int("".join(str(d) for d in tokens[2:-1]))
      sign = int("".join([tokens[1]]+['1']))
      mantissa  = sign*number
      # text = by.decode("utf-8",errors='replace')
      exp = self.get_exp_num(tokens[-1])
      if exp[1:].startswith('-'):
        exp = 10**(-int(exp[2:]))
      else: exp = 10**(int(exp[1:]))
      return mantissa*exp
    except ValueError:
      return 'Prediction out of range'
