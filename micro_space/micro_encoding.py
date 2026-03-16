from collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype_norm = namedtuple('Genotype', 'normal normal_concat')
Genotype_redu = namedtuple('Genotype', 'reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'sep_conv_7x7',
    'conv_7x1_1x7',
]

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2


def convert_cell(cell_bit_string):
    # convert cell bit-string to genome
    tmp = [cell_bit_string[i:i + 2] for i in range(0, len(cell_bit_string), 2)]
    return [tmp[i:i + 2] for i in range(0, len(tmp), 2)]


def convert(bit_string):
    # convert network bit-string (norm_cell + redu_cell) to genome
    norm_gene = convert_cell(bit_string[:len(bit_string)//2])
    redu_gene = convert_cell(bit_string[len(bit_string)//2:])
    return [norm_gene, redu_gene]


def decode_cell(genome, norm=True):

    cell, cell_concat = [], list(range(2, len(genome)+2))
    for block in genome:
        for unit in block:
            cell.append((PRIMITIVES[unit[0]], unit[1]))
            if unit[1] in cell_concat:
                cell_concat.remove(unit[1])

    if norm:
        return Genotype_norm(normal=cell, normal_concat=cell_concat)
    else:
        return Genotype_redu(reduce=cell, reduce_concat=cell_concat)


def decode(genome, steps=6, multiplier=6):
    # decodes genome to architecture
    normal_cell = genome[0]
    reduce_cell = genome[1]

    # intern nodes are concatenad, except the input nodes (0,1)
    normal, normal_concat = [], range(2 + steps - multiplier, steps + 2)
    reduce, reduce_concat = [], range(2 + steps - multiplier, steps + 2)

    for block in normal_cell:
        for unit in block:
            normal.append((PRIMITIVES[unit[0]], unit[1]))

    for block in reduce_cell:
        for unit in block:
            reduce.append((PRIMITIVES[unit[0]], unit[1]))

    return Genotype(
        normal=normal, normal_concat=normal_concat,
        reduce=reduce, reduce_concat=reduce_concat
    )


def compare_cell(cell_string1, cell_string2):
    cell_genome1 = convert_cell(cell_string1)
    cell_genome2 = convert_cell(cell_string2)
    cell1, cell2 = cell_genome1[:], cell_genome2[:]

    for block1 in cell1:
        for block2 in cell2:
            if block1 == block2 or block1 == block2[::-1]:
                cell2.remove(block2)
                break
    if len(cell2) > 0:
        return False
    else:
        return True


def compare(string1, string2):

    if compare_cell(string1[:len(string1)//2],
                    string2[:len(string2)//2]):
        if compare_cell(string1[len(string1)//2:],
                        string2[len(string2)//2:]):
            return True

    return False
