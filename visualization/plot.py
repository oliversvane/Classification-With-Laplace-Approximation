import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calc_bins(preds):
  # Assign each prediction to a bin
  num_bins = 10
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

  return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)

  return ECE, MCE

def draw_reliability_graph(preds,name):
  ECE, MCE = get_metrics(preds)
  bins, _, bin_accs, _, _ = calc_bins(preds)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.gca()

  # x/y limits
  ax.set_xlim(0, 1.05)
  ax.set_ylim(0, 1)

  # x/y labels
  plt.xlabel('Confidence')
  plt.ylabel('Accuracy')

  # Create grid
  ax.set_axisbelow(True) 
  ax.grid(color='gray', linestyle='dashed')

  # Error bars
  plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

  # Draw bars and identity line
  plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
  plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

  # Equally spaced axes
  plt.gca().set_aspect('equal', adjustable='box')

  # ECE and MCE legend
  ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
  MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
  plt.legend(handles=[ECE_patch, MCE_patch])

  #plt.show()
  
  plt.savefig(name+'calibrated_network.png', bbox_inches='tight')

#draw_reliability_graph(preds)


def cor_var(y):
    variances=[]
    coordinates=[]
    for i in range(len(y)):
        variances.append((np.var([x[0] for x in y[i]]),(np.var([x[1] for x in y[i]]))))
        coordinates.append([np.mean([x[0] for x in y[i]]),(np.mean([x[1] for x in y[i]]))])
    return coordinates,variances

def Ellipse_plot(cor,var,cls):
    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    ax.axis('equal')
    ax.set(xlim=[0, 5], ylim=[0, 5])
    for i in range(len(cor)):
        if cls[i] == True:
            b="b"
            ell = Ellipse(xy=cor[i], width=var[i][0], height=var[i][1], angle=0,
                        edgecolor=b, lw=2, facecolor='none')
            ax.scatter(cor[i][0], cor[i][1],s=3, c=b)
            ax.add_artist(ell)
        else:
            ell = Ellipse(xy=cor[i], width=var[i][0], height=var[i][1], angle=0,
            edgecolor="r", lw=2, facecolor='none')
            ax.scatter(cor[i][0], cor[i][1],s=3, c="r") 
            ax.add_artist(ell)
        my_labels = {"x1" : "_nolegend_", "x2" : "_nolegend_"}   

    #ax.legend()

    plt.show()

def dist_plot(var):
    dist_data_x1 = [x[0] for x in var]
    dist_data_x2 = [x[1] for x in var]
    df = pd.DataFrame(list(zip(dist_data_x1, dist_data_x2,cls)),
                columns =['x', 'y','dist'])
    sns.displot(df, x="x", hue="dist", kind="kde", multiple="stack")



def project_plot(col,fake_labels):
    #tester=["Hat","Shoe","Dress","Hat","Shoe","Dress","Hat","Shoe","Dress","jacket"]
    fig, ax = plt.subplots()
    reducer = umap.UMAP()
    embedding=reducer.fit_transform(col)
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        #c=[sns.color_palette()[x] for x in pd.Series(fake_labels).map({1:0, 2:1, 3:2,4:3, 5:4, 6:5,7:6, 8:7, 9:8,10:9})])
        c=fake_labels,cmap=plt.cm.get_cmap('gist_rainbow', 9))
    plt.gca().set_aspect('equal', 'datalim')
    cbar=plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    #cbar.set_ticklabels([tester]) VIRKER IKKE  
    #cbar = plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10)), orientation='vertical')
    plt.title('Fashion Mnist', fontsize=24)
    plt.show()