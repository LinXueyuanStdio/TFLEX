"""
@date: 2021/10/27
@description: 交互式表达式解释器，输入 query 输出 answer
"""
import cmd
import sys
import traceback
from typing import List, Tuple

import expression
from ComplexTemporalQueryData import build_mapping_simple


class ExpressionInterpreter(cmd.Cmd):
    """
    Interactive command line interpreter that applies the expression line parser
    to the provided input.
    """

    def __init__(self, triples_ids: List[Tuple[int, int, int, int]]):
        cmd.Cmd.__init__(self)
        self.prompt = '>> '
        self.train_triples_ids = triples_ids
        entities_ids = set()
        relations_ids = set()
        timestamps_ids = set()
        for s, r, o, t in self.train_triples_ids:
            entities_ids.add(s)
            relations_ids.add(r)
            entities_ids.add(o)
            timestamps_ids.add(t)
        self.entities_ids = list(entities_ids)
        self.relations_ids = list(relations_ids)
        self.timestamps_ids = list(timestamps_ids)

        max_relation_id = len(self.relations_ids)
        relations_ids_with_reverse = self.relations_ids + [r + max_relation_id for r in self.relations_ids]

        def append_reverse(triples):
            nonlocal max_relation_id
            res = []
            for s, r, o, t in triples:
                res.append((s, r, o, t))
                res.append((o, r + max_relation_id, s, t))
            return res

        train_triples_ids = append_reverse(self.train_triples_ids)
        train_sro_t, train_sor_t, train_srt_o, train_str_o, \
        train_ors_t, train_trs_o, train_tro_s, train_rst_o, \
        train_rso_t, train_t_sro, train_o_srt = build_mapping_simple(train_triples_ids)
        self.parser = expression.SamplingParser(self.entities_ids, relations_ids_with_reverse, self.timestamps_ids,
                                                train_sro_t, train_sor_t, train_srt_o, train_str_o,
                                                train_ors_t, train_trs_o, train_tro_s, train_rst_o,
                                                train_rso_t, train_t_sro, train_o_srt)

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
    s, r, o, t = 1, 1, 1, 1
    triples_ids = [(s, r, o, t)]
    ExpressionInterpreter(triples_ids).cmdloop()


if __name__ == '__main__':
    main()
