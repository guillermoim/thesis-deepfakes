from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


# CIFAR dataset

pdarts = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0),('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3',0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

# Neural texture dataset

nt_da_policy_prove_and_error = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('skip_connect', 2), ('dil_conv_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))


nt_da_policy_1 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('avg_pool_3x3', 1), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 3), ('sep_conv_3x3', 1), ('skip_connect', 4)], reduce_concat=range(2, 6))


nt_da_policy_2 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1), ('skip_connect', 3), ('dil_conv_5x5', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))


nt_da_policy_3 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

nt_da_policy_4 = Genotype(normal=[('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 3), ('dil_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))


nt_da_policy_5 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 3), ('dil_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

# GENOTYPES

GENOTYPES = {

	'cifar': pdarts,

	0: nt_da_policy_prove_and_error,

	1: nt_da_policy_1,

	2: nt_da_policy_2,

	3: nt_da_policy_3,

	4: nt_da_policy_4,

	5: nt_da_policy_5,
	
}
