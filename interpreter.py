"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2021/10/27
@description: null
"""

from __future__ import print_function

import cmd
import math
import sys
import traceback

import expression


class ExpressionInterpreter(cmd.Cmd):
    """
    Interactive command line interpreter that applies the expression line parser
    to the provided input.
    """

    def __init__(self):
        cmd.Cmd.__init__(self)
        self.prompt = '>> '

        # you can extend variables and functions by yourselves!
        variables = {
            'pi': math.pi,
            'e': math.e
        }
        functions = {
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'pow': math.pow,
            'sqrt': math.sqrt
        }
        self.parser = expression.ExpressionParser(variables=variables, functions=functions, assignment=True)

    def default(self, line):
        try:
            output = self.parser.parse(line)
            if output is not None:
                self.stdout.write(str(output) + '\n')

            variables = self.parser.variables
            variables.update(self.parser.modified_variables)
            self.parser.variables = variables
        except SyntaxError:
            traceback.print_exc(0)

    def do_quit(self, line):
        """
        Exit the interpreter.
        """

        if line != '' and line != '()':
            self.stdout.write(line + '\n')
        self._quit()

    @staticmethod
    def _quit():
        sys.exit(1)


def main():
    ExpressionInterpreter().cmdloop()


if __name__ == '__main__':
    main()
