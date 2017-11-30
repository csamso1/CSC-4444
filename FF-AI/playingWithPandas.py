import pandas as pd

def main():
	data_array = [12, 22, 33]
	index_array = [1, 2, 3]
	columns = [5, 5, 5]
	test_df = pd.DataFrame({'name' : ['clayton', 'leighton', 'ricky', 'amy', 'christy', 'ted'], 'values' : [2, 3, 5, 7, 11, 69]})
	test_df_2 = pd.DataFrame({'name' : ['clayton', 'leighton', 'ricky', 'amy', 'christy', 'ted'], 'values' :[4, 6, 8, 10, 12, 69]})
	#test_data_frame = pd.DataFrame(data_array, index=index_array, columns=columns)
	print('test_df\n')
	print(test_df.head())
	print('test_df_2\n')
	print(test_df_2.head())
	merged_DF = pd.merge(test_df, test_df_2, on='name')
	#creates redundant colums: left_index=True, right_index=True
	#test_df.join(test_df_2)
	print('merged_DF: \n')
	print(merged_DF)
	print('removing all but ted')
	merged_DF = merged_DF[merged_DF.name == 'ted']
	print(merged_DF)
main()