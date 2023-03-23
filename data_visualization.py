import matplotlib.pyplot as plt
import itertools
import os
import math


class DataPlot:
    """Class to manage the data visualization for current exercise"""

    def __init__(self):
        self.fig_width = 20
        self.fig_height = 10

    def pie_plot(self, df, column_target):
        """Create pie plot for the distribution of the column called column_target in the dataframe df"""
        fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
        weights = df.groupby([column_target]).size()
        labels = weights.index.values
        labels_count = [str(labels[i]) + ' (' + str(weights[i]) + ')' for i in range(len(labels))]
        labels_pct = [str(labels[i]) + ' (' + str(round(100*weights[i]/sum(weights), 1)) + '%)'
                      for i in range(len(labels))]
        explode = [0.1] * len(weights)
        ax.pie(x=weights, explode=explode, labels=labels_count, autopct='%1.1f%%', shadow=True,
               textprops={'fontsize': 16})
        ax.set_title('PASSENGERS TRANSPORTED DISTRIBUTION', fontweight='bold', fontsize=24)
        ax.legend(labels=labels_pct, loc='upper left', bbox_to_anchor=(1.1, 1), fancybox=True, shadow=True,
                  prop={'size': 16})
        ax.grid(visible=True)
        fig.tight_layout()
        plt.savefig('Passengers transported distribution.png', bbox_inches='tight')
        plt.close()

    def combined_features(self, df, fcat, column_target):
        feat_cat = fcat.copy()
        if column_target in feat_cat:
            feat_cat.remove(column_target)
        combs = list(itertools.combinations(feat_cat, 2))
        for i, comb in enumerate(combs):
            list_feat = list(comb)
            fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
            feats = list_feat + [column_target]
            df.groupby(feats).size().unstack().plot(kind='bar', ax=ax)
            ax.grid(visible=True)
            ax.tick_params(axis='both', labelsize=10, labelrotation=70)
            ax.set_ylabel('FREQUENCY', fontsize=14, fontweight='bold')
            ax.set_xlabel('FEATURE MAGNITUDE ' + str(list_feat[0].upper()) + ' and ' + str(list_feat[1].upper()),
                          fontsize=14, fontweight='bold')
            ax.set_title('COMBINED PLOT FOR FEATURE CORRELATION ASSESSMENT BETWEEN ' + str(list_feat[0].upper()) +
                         ', ' + str(list_feat[1].upper()) + ' AND ' + str(column_target.upper()), fontsize=20,
                         fontweight='bold')
            if column_target == 'Transported':
                plt.savefig(os.getcwd() + '\\TargetPlots\\' + str(column_target) + '_' + str(list_feat[0]) + '_' +
                            str(list_feat[1]) + '_Combined_Plot.png', bbox_inches='tight')
            else:
                plt.savefig(os.getcwd() + '\\CombinedPlots\\' + str(column_target) + '_' + str(list_feat[0]) + '_' +
                            str(list_feat[1]) + '_Combined_Plot.png', bbox_inches='tight')
            plt.close()

    def cat_features(self, df, feat_cat, column_target):
        """Plot categorical features"""
        fcat = feat_cat.copy()
        if 'Num' in fcat:
            fcat.remove('Num')
        if 'ShipLocation' in fcat:
            fcat.remove('ShipLocation')
        ncolumns = 3
        plots = len(fcat)
        if column_target in fcat:
            backup = fcat[-1]
            fcat[-1] = column_target
            fcat[fcat.index(column_target)] = backup
            plots -= 1
        fig, axes = plt.subplots(math.ceil(plots / ncolumns), ncolumns, figsize=(self.fig_width, self.fig_height))
        spare_axes = ncolumns - plots % ncolumns
        if spare_axes == ncolumns:
            spare_axes = 0
        for axis in range(ncolumns - 1, ncolumns - 1 - spare_axes, -1):
            if (math.ceil(plots / ncolumns) - 1) == 0:
                fig.delaxes(axes[axis])
            else:
                fig.delaxes(axes[math.ceil(plots / ncolumns) - 1, axis])
        ax = axes.ravel()
        for i in range(plots):
            df.groupby([fcat[i], column_target]).size().unstack().plot(kind='bar', ax=ax[i])
            ax[i].tick_params(axis='x', labelrotation=10)
            ax[i].set_ylabel('FREQUENCY', fontsize=10, fontweight='bold')
            ax[i].set_xlabel('FEATURE MAGNITUDE ' + fcat[i].upper(), fontsize=10, fontweight='bold')
            ax[i].set_title(column_target.upper() + ' DISTRIBUTION GROUPED BY ' + fcat[i].upper(),
                            fontweight='bold', fontsize=12)
            ax[i].grid(visible=True)
        fig.tight_layout()
        plt.savefig(column_target.title() + ' distribution per features.png', bbox_inches='tight')
        plt.close()

    def num_features(self, df, feat_num, binary_feat, bins_width):
        """Plot a histogram for each numerical feature"""
        fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
        output0 = df[df[binary_feat] == False][feat_num]
        output1 = df[df[binary_feat] == True][feat_num]
        max_val = max(max(output0), max(output1))
        bins = list(range(bins_width, int(max_val), bins_width))
        ax.set_xticks(bins)
        ax.set_xticklabels(bins, rotation=70)
        ax.grid(visible=True)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylabel('FREQUENCY', fontsize=14, fontweight='bold')
        ax.set_xlabel('FEATURE MAGNITUDE ' + feat_num.upper(), fontsize=14, fontweight='bold')
        ax.set_title('HISTOGRAM FOR ' + feat_num.upper() + ' FEATURE GROUPED BY TARGET', fontsize=20, fontweight='bold')
        ax.hist(output0, histtype='stepfilled', bins=bins, alpha=0.25, color='b', lw=0, label=binary_feat + '=0')
        ax.hist(output1, histtype='stepfilled', bins=bins, alpha=0.25, color='r', lw=0, label=binary_feat + '=1')
        ax.legend()
        fig.tight_layout()
        plt.savefig(binary_feat.title() + ' distribution grouped by ' + feat_num + '.png', bbox_inches='tight')
        plt.close()

