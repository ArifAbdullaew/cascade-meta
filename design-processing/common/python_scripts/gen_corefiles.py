# Copyright 2023 Flavien Solt, ETH Zurich.
# Licensed under the General Public License, Version 3.0, see LICENSE for details.
# SPDX-License-Identifier: GPL-3.0-only

# sys.argv[1]: source template core file
# sys.argv[2]: destination template core file

import os
import re
import sys

if __name__ == "__main__":
    src_filename = sys.argv[1]
    tgt_filename = sys.argv[2]

    with open(src_filename, "r") as f:
        content = f.read()

    # Replace $VARNAME with environment values
    pattern_dollar = r'\$([A-Za-z_]+[A-Za-z0-9_]*)'
    matches = re.findall(pattern_dollar, content)
    for match in matches:
        env_var_value = os.environ.get(match, '')
        content = content.replace('$' + match, env_var_value)

    # Extract design_name from folder name
    design_name = os.path.basename(os.path.abspath(os.path.dirname(tgt_filename)))

    # Replace {{design_name}} with actual design name
    content = re.sub(r'\{\{\s*design_name\s*\}\}', design_name, content)

    with open(tgt_filename, "w") as f:
        f.write(content)

