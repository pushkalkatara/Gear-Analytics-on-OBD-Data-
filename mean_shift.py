import os
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# Function to Clean Data
'''
Check whether Engine ON df['rpm'] > 0
Check whether Vehicle Moving df[speed] > 0
'''
def clean(df):
	original_df = df.dropna(axis = 0, how = 'any')
	is_moving = original_df['speed'] > 0.
	is_on = original_df['rpm'] > 0.
	is_loaded = original_df['load'] > 0.
	mask = is_moving & is_on & is_loaded
	return original_df[mask]

#Function to Visualize Data
def visualize_df(df):
    df.plot(x = 'rpm', y = 'speed', kind = 'scatter', alpha = .2)
    plt.suptitle('RPM vs Car Speed')
    plt.show()
    return

#Function to convert to Vehicle/Engine Speed Ratio DataFrame
def make_polar_df(df):
	rpm = df['rpm']
	speed = df['speed']
	phi = np.arctan2(speed, rpm)
	phi_mag = np.sqrt(rpm ** 2 + speed ** 2)
	return pd.DataFrame({'phi' : phi, 'rho' : phi_mag})

#Mean Shift
def mean_shift_cluster(df, feature):
	x = df[feature]
	X = np.array(zip(x,np.zeros(len(x))), dtype=np.float)
	bandwidth = estimate_bandwidth(X, quantile=0.1)
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	ms.fit(X)
	return ms

#To Filter Edges
def filter_neutral(df, ms):
	cluster_centers = ms.cluster_centers_[:,0]
	neutral_label = np.where(cluster_centers == cluster_centers.min())[0][0]
	ms_labels = ms.labels_
	neutral_mask = ms_labels == neutral_label
	return df[~neutral_mask]

def visualize_polar_df(polar_df):
	polar_df.plot(x = 'rho', y = 'phi', kind = 'scatter', alpha = .02)
	plt.suptitle('PHI (vehicle/engine speed ratio) vs. RHO (Magnitude)')
	plt.show()
	return

def label_gears(polar_df, k, gears):
	cluster_centers = k.cluster_centers_[:,0]
	label_dict = dict(enumerate(cluster_centers))
	label_dict = {v:r for r,v in label_dict.iteritems()}
	labels = k.labels_
	label_counts = np.bincount(labels)
	thresh = sorted(label_counts, reverse = True)[gears - 1]
	wanted_labels = np.where(label_counts >= thresh)[0]
	rev_label_dict = {v:r for r,v in label_dict.iteritems()}
	gear_dict = dict(enumerate(sorted([rev_label_dict[label] for label in wanted_labels]),1))
	print(gear_dict)
	gear_dict = {v:r for r,v in gear_dict.iteritems()}
	gear_lines = np.array(gear_dict.keys())
	translate_dict = {}
	for r,v in label_dict.iteritems():
		label = v
		if gear_dict.has_key(r):
			gear_label = gear_dict[r]
		else:
			gear_label = 0
		translate_dict[label] = gear_label
	polar_df['label'] = labels
	polar_df['gear'] = polar_df['label'].replace(translate_dict)
	return polar_df.drop('label', axis = 1), np.sort(gear_dict.keys())

def visualize_gears(merged, n_gears):
	lm = sns.lmplot(x = 'rho',
		y = 'phi',
		data = merged,
		hue = 'gear',
		size = 6,
		aspect = 1.6,
		fit_reg = False,
		# palette ='cubehelix',
		hue_order = range(n_gears + 1)[::-1],
		scatter_kws = {'marker' : 'D',
						'alpha' : 0.3,
						's' : 30
						})
	axes = lm.axes
	axes[0,0].set_ylim(0,)
	plt.show()
	return

def main():
    #Visualize Original Dataset
	sns.set_style('whitegrid')
	original_df = pd.read_csv('147.csv')
	clos = ['speed','rpm']
	visualize_df(original_df[clos])

	#Filter Dataset
	moving_df = clean(original_df)
	rpm_ms = mean_shift_cluster(moving_df, 'rpm')
	neutral_df = filter_neutral(moving_df, rpm_ms)
	neutral_df.to_csv('147_filtered.csv')
	polar_df = make_polar_df(neutral_df)
	visualize_polar_df(polar_df)
	clos = ['phi','rho']

	#Mean Shift on basis of phi.
	phi_ms = mean_shift_cluster(polar_df, 'phi')
	polar_df, lines = label_gears(polar_df, phi_ms, 5)
	polar_df.to_csv('147_gear.csv')
	visualize_polar_df(polar_df)
	merged = moving_df.merge(polar_df, how = 'left', left_index = True, right_index = True)
	merged.loc[:,'gear'] = merged['gear'].fillna(0).astype(int)
	visualize_gears(merged, 6)
	plt.plot(polar_df)
	plt.show()
if __name__ == '__main__':
	main()
