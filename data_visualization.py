import matplotlib.pyplot as plt
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

    def cat_features(self, df, feat_cat, column_target, multi=0):
        """Plot categorical features
        - multi = 0 applies each categorical features vs column_target in an individual plot
        - multi = 1 applies all categorical features vs column_target in the same plot"""
        if multi == 0:
            ncolumns = 2
            nlen = len(feat_cat)
            tag = 'Categorical features.png'
        else:
            ncolumns = 1
            nlen = 1
            tag = 'Multi categorical features.png'
        fig, axes = plt.subplots(math.ceil(nlen / ncolumns), ncolumns, figsize=(self.fig_width, self.fig_height))
        spare_axes = ncolumns - nlen % ncolumns
        if spare_axes == ncolumns:
            spare_axes = 0
        for axis in range(ncolumns - 1, ncolumns - 1 - spare_axes, -1):
            fig.delaxes(axes[math.ceil(nlen / ncolumns) - 1, axis])
        if multi == 0:
            ax = axes.ravel()
            for i, column in enumerate(feat_cat):
                df.groupby([column, column_target]).size().unstack(0).plot(kind='bar', ax=ax[i])
                ax[i].tick_params(axis='x', labelrotation=60)
                ax[i].set_ylabel('FREQUENCY', fontsize=12, fontweight='bold')
                ax[i].set_xlabel('TRANSPORTED', fontsize=12, fontweight='bold')
                ax[i].set_title('TRANSPORTED DISTRIBUTION GROUPED BY ' + column.upper(), fontweight='bold', fontsize=14)
                ax[i].grid(visible=True)
        else:
            feat_cat.append(column_target)
            df.groupby(feat_cat).size().unstack().plot(kind='bar', ax=axes)
            axes.tick_params(axis='x', labelrotation=60)
            axes.set_ylabel('FREQUENCY', fontsize=16, fontweight='bold')
            axes.set_xlabel(feat_cat[:-1], fontsize=16, fontweight='bold')
            axes.set_title('TRANSPORTED DISTRIBUTION GROUPED BY ' + str(feat_cat[:-1]), fontweight='bold', fontsize=24)
            axes.grid(visible=True)
        fig.tight_layout()
        plt.savefig(tag, bbox_inches='tight')
        plt.close()

    def num_features(self, df, feat_num, column_target):
        """Plot a histogram for each numerical feature"""
        ncolumns = 2
        nlen = len(feat_num)
        fig, axes = plt.subplots(math.ceil(nlen / ncolumns), ncolumns, figsize=(self.fig_width, self.fig_height))
        spare_axes = ncolumns - nlen % ncolumns
        if spare_axes == ncolumns:
            spare_axes = 0
        for axis in range(ncolumns - 1, ncolumns - 1 - spare_axes, -1):
            fig.delaxes(axes[math.ceil(nlen / ncolumns) - 1, axis])
        ax = axes.ravel()
        for i, column in enumerate(feat_num):
            output0 = df[df[column_target] == False][column]
            output1 = df[df[column_target] == True][column]
            max_val = max(max(output0), max(output1))
            bins = list(range(0, int(max_val), max(1, int(max_val / 25))))
            ax[i].hist(output0, histtype='stepfilled', bins=bins, alpha=0.25, color="#0000FF", lw=0)
            ax[i].hist(output1, histtype='stepfilled', bins=bins, alpha=0.25, color='#FF0000', lw=0)
            ax[i].set_title(column.upper(), fontsize=12, y=1.0, pad=-14, fontweight='bold')
            ax[i].grid(visible=True)
            ax[i].tick_params(axis='both', labelsize=10)
            ax[i].set_ylabel('FREQUENCY', fontsize=10)
            ax[i].set_xlabel('FEATURE MAGNITUDE', fontsize=10)
            ax[i].legend(['Transported=False', 'Transported=True'], loc="best")
        fig.suptitle('HISTOGRAM FOR NUMERICAL FEATURES GROUPED BY TRANSPORTED VALUE', fontsize=18, fontweight='bold')
        fig.tight_layout()
        plt.savefig('Numerical features.png', bbox_inches='tight')
        plt.close()
