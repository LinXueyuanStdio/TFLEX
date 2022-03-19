"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2021/10/27
@description: null
"""
import cmd
import random
import sys
import traceback
from collections import defaultdict

import expression


class ExpressionInterpreter(cmd.Cmd):
    """
    Interactive command line interpreter that applies the expression line parser
    to the provided input.
    """

    def __init__(self):
        cmd.Cmd.__init__(self)
        self.prompt = '>> '
        srt2o = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: random.sample(list(range(5)), 3))))
        sro2t = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: random.sample(list(range(5)), 3))))

        def random_ids():
            return random.sample(list(range(5)), 3)

        for s in random_ids():
            for r in random_ids():
                for o in random_ids():
                    sro2t[s][r][o] = random_ids()
                for t in random_ids():
                    srt2o[s][r][t] = random_ids()
        self.parser = expression.SamplingParser(
            list(range(5)),
            list(range(5)),
            list(range(5)),
            srt2o, sro2t,
        )

    def default(self, line):
        try:
            output = self.parser.eval(line)
            if output is not None:
                self.stdout.write(str(output) + '\n')
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
