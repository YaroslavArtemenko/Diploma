# import libraries
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, AgglomerativeClustering
from sklearn.manifold import TSNE

from Forms.DataCreate import DataCreateForm
from Forms.DataEdit import DataFormEdit
from Forms.SearchForm import SearchForm

# replace E:/to-postgres.csv on any import name file
# \copy public.rate (country_name, index_value, usage_value, continent_name) FROM 'E:/to-postrgres.csv' DELIMITER ',' \
# CSV HEADER ENCODING 'UTF8' QUOTE '\"' ESCAPE '''';


db = SQLAlchemy()

#create table
class Rate(db.Model):
    __tablename__ = 'rate'
    country_name = db.Column(db.String(20), primary_key=True)
    index_value = db.Column(db.Float(10))
    usage_value = db.Column(db.Float(10))
    continent_name = db.Column(db.String(20))

#connect to DataBase
def create_app():
    """
    initialize Flask app
    :return: app
    """
    app = Flask(__name__)
    app.secret_key = 'key'

    # TODO: read from config
    ENV = 'dev'

    if ENV == 'dev':
        app.debug = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:Trouble228@localhost/Rate'
    else:
        app.debug = False
        app.config['SECRET_KEY'] = 'laba2artemenko'
        app.config['SQLALCHEMY_DATABASE_URI'] = "postgres://bbehxvtelkevci:19f969a6e8c4ba12b641616788d67e03b2db997bc" \
                                                "6727471ff5f560ce5ba2d18@ec2-54-217-204-34.eu-west-1.compute.amazona" \
                                                "ws.com:5432/d4n91kp6vskkic"

    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    app.app_context().push()
    db.create_all()
    return app


app = create_app()


#start page
@app.route('/')
def root():
    return render_template('index.html')


#output all data
@app.route('/all', methods=['GET'])
def all_data():
    result = db.session.query(Rate).order_by(Rate.country_name).all()
    print(result)
    return render_template('all_data.html', result=result)


#edit data
@app.route('/edit_data/<string:country>', methods=['GET', 'POST'])
def edit_data(country):
    form = DataFormEdit()
    result = db.session.query(Rate).filter(Rate.country_name == country).one()

    if request.method == 'GET':

        form.country_name.data = result.country_name
        form.index_value.data = result.index_value
        form.usage_value.data = result.usage_value
        form.continent_name.data = result.continent_name

        return render_template('edit_data.html', form=form, form_name=country)
    elif request.method == 'POST':
        if form.validate() and form.check_index() and form.check_usage():
            result.country_name = form.country_name.data
            result.index_value = form.index_value.data
            result.usage_value = form.usage_value.data
            result.continent_name = form.continent_name.data

            db.session.commit()
            return redirect('/all')
        else:
            if not form.check_index():
                form.index_value.errors = ['should be > 0']
            if not form.check_usage():
                form.usage_value.errors = ['should be > 0']
            return render_template('edit_data.html', form=form)


#search with criterion
@app.route('/search', methods=['POST', 'GET'])
def search():
    form = SearchForm()

    if request.method == 'POST':
        if form.type_field.data == 'country_name':
            res = db.session.query(Rate).filter(Rate.country_name == form.search_value.data).all()
        elif form.type_field.data == 'continent_name':
            res = db.session.query(Rate).filter(Rate.continent_name == form.search_value.data).all()

        return render_template('search_result.html', countries=res)
    else:
        return render_template('search.html', form=form)


#create new data
@app.route('/create_data', methods=['POST', 'GET'])
def create_country():
    form = DataCreateForm()
    try:
        result = db.session.query(Rate).filter(Rate.country_name == form.country_name.data).one()
        if result != 0:
            return render_template('create_data.html', contest_name="Data exist", form=form)
    except Exception:
        pass
    if request.method == 'POST':
        if form.validate() and form.check_index() and form.check_usage():
            new_data = Rate(
                country_name=form.country_name.data,
                index_value=form.index_value.data,
                usage_value=form.usage_value.data,
                continent_name=form.continent_name.data
            )
            db.session.add(new_data)
            db.session.commit()
            return redirect('/all')
        else:
            if not form.check_index():
                form.index_value.errors = ['should be > 0']
            if not form.check_usage():
                form.usage_value.errors = ['should be > 0']
            return render_template('create_data.html', form=form)
    elif request.method == 'GET':
        return render_template('create_data.html', form=form)


#delete data
@app.route('/delete_data/<string:country>', methods=['GET', 'POST'])
def delete_data(country):
    result = db.session.query(Rate).filter(Rate.country_name == country).one()

    db.session.delete(result)
    db.session.commit()

    return redirect('/all')


RANDOM_STATE = 12
df = pd.DataFrame()
x_tsne = pd.DataFrame()
X = pd.DataFrame()
y = None
col_1 = 'column_1'
col_2 = 'column_2'
cluster_num = 0


#read data for analysis
def get_latest_data():
    global df, x_tsne, col_1, col_2, RANDOM_STATE, X, y, cluster_num

    df = pd.DataFrame()
    for country, index, usage, continent in db.session.query(Rate.country_name, Rate.index_value, Rate.usage_value,
                                                             Rate.continent_name):
        df = df.append({"country": country, "human_index": index, "usage": usage, "continent": continent},
                       ignore_index=True)

    col_1 = 'human_index'
    col_2 = 'usage'
    target = 'continent'
    X = df[[col_1, col_2]]
    RANDOM_STATE = 12
    tsne = TSNE(random_state=RANDOM_STATE)
    data_norm = preprocessing.scale(X)
    x_tsne = tsne.fit_transform(data_norm)
    le = preprocessing.LabelEncoder()
    le.fit(df[target])
    y = le.transform(df[target])
    cluster_num = np.max(y) + 1


#create scatter-matrix
@app.route('/correlation', methods=['GET', 'POST'])
def correlation():
    get_latest_data()
    plt.cla()
    plt.clf()
    _ = scatter_matrix(df, alpha=0.5, figsize=(10, 10))

    plt.savefig('static/images/matrix.png')
    # plt.figure(figsize=(6.4, 4.8))

    return render_template('result_1.html')

#create t-SNE
@app.route('/tsne', methods=['GET', 'POST'])
def tsne():
    get_latest_data()
    plt.cla()
    plt.clf()
    plt.scatter(df[col_1], df[col_2], c=y, edgecolor='none', alpha=0.7, s=40,
                cmap=plt.cm.get_cmap('nipy_spectral', cluster_num))
    plt.colorbar()
    plt.xlabel(col_1)
    plt.ylabel(col_2)
    plt.savefig('static/images/human.png')

    plt.cla()
    plt.clf()
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y,
                edgecolor='none', alpha=0.7, s=40,
                cmap=plt.cm.get_cmap('nipy_spectral', cluster_num))
    plt.colorbar()
    plt.title("TSNE for {} random state".format(RANDOM_STATE))
    plt.xlabel(col_1)
    plt.ylabel(col_2)
    plt.savefig('static/images/tsne.png')

    return render_template('result_2.html')


#create method K-Means
@app.route('/kmeans', methods=['GET', 'POST'])
def kmeans():
    get_latest_data()
    # start kMeans algo
    plt.cla()
    plt.clf()
    inertia = []
    for k in range(1, cluster_num):
        kmeans = KMeans(n_clusters=k, random_state=13).fit(x_tsne)
        inertia.append(np.sqrt(kmeans.inertia_))

    plt.plot(range(1, cluster_num), inertia, marker='s')
    plt.title("k-Means")
    plt.xlabel('$k$')
    plt.ylabel('$J(C_k)$')
    plt.savefig('static/images/kmeans.png')


    plt.cla()
    plt.clf()

    kmeans = KMeans(n_clusters=5, max_iter=300, n_init=10, random_state=13)
    pred_y = kmeans.fit_predict(x_tsne)
    plt.scatter(
        x_tsne[pred_y == 0, 0], x_tsne[pred_y == 0, 1],
        s=50, c='lightgreen',
        marker='o', edgecolor='black',
        label='cluster 1'
    )

    plt.scatter(
        x_tsne[pred_y == 1, 0], x_tsne[pred_y == 1, 1],
        s=50, c='orange',
        marker='o', edgecolor='black',
        label='cluster 2'
    )

    plt.scatter(
        x_tsne[pred_y == 2, 0], x_tsne[pred_y == 2, 1],
        s=50, c='lightblue',
        marker='o', edgecolor='black',
        label='cluster 3'
    )

    plt.scatter(
        x_tsne[pred_y == 3, 0], x_tsne[pred_y == 3, 1],
        s=50, c='green',
        marker='o', edgecolor='black',
        label='cluster 4'
    )


    plt.scatter(
        x_tsne[pred_y == 4, 0], x_tsne[pred_y == 4, 1],
        s=50, c='blue',
        marker='o', edgecolor='black',
        label='cluster 5'
    )

    plt.scatter(
        kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
        s=300, marker='o',
        c='red', edgecolor='black',
        label='centroids'
    )
    plt.title("k-Means")
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.savefig('static/images/kmeans_2.png')

    return render_template('result_3.html')


#create Spectral Clustering
@app.route('/spectral', methods=['GET', 'POST'])
def spectral():
    get_latest_data()
    plt.cla()
    plt.clf()
    # build spectral clustering
    spectral = SpectralClustering(n_clusters=cluster_num, random_state=1, affinity='nearest_neighbors').fit(x_tsne)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y,
                edgecolor='none', alpha=0.7, s=40,
                cmap=plt.cm.get_cmap('nipy_spectral', cluster_num))
    plt.title("Real data vs Spectral clustering")
    ax2.scatter(x_tsne[:, 0], x_tsne[:, 1], c=spectral.labels_,
                edgecolor='none', alpha=0.7, s=40,
                cmap=plt.cm.get_cmap('nipy_spectral', cluster_num))
    plt.savefig('static/images/spectral.png')
    return render_template('result_4.html')


#create Agglomerative Clustering
@app.route('/agglomerative', methods=['GET', 'POST'])
def agglomerative():
    get_latest_data()
    plt.cla()
    plt.clf()
    agglomerative = AgglomerativeClustering(n_clusters=cluster_num).fit(x_tsne)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, edgecolor='none', alpha=0.7, s=40,
                cmap=plt.cm.get_cmap('nipy_spectral', cluster_num))
    plt.title("Real data vs agglomerative clustering")
    ax2.scatter(x_tsne[:, 0], x_tsne[:, 1], c=agglomerative.labels_, edgecolor='none', alpha=0.7, s=40,
                cmap=plt.cm.get_cmap('nipy_spectral', cluster_num))
    plt.savefig('static/images/agglo.png')
    # plt.show()

    return render_template('result_5.html')


#generate Comparison Table
@app.route('/comparison_table', methods=['GET', 'POST'])
def comparison_table():
    get_latest_data()
    algorithms = []
    algorithms.append(KMeans(n_clusters=cluster_num, random_state=1))
    algorithms.append(AffinityPropagation())
    algorithms.append(SpectralClustering(n_clusters=cluster_num, random_state=1,
                                         affinity='nearest_neighbors'))
    algorithms.append(AgglomerativeClustering(n_clusters=cluster_num))

    data = []
    for algo in algorithms:
        algo.fit(x_tsne)
        data.append(({
            'ARI': metrics.adjusted_rand_score(y, algo.labels_),
            'AMI': metrics.adjusted_mutual_info_score(y, algo.labels_),
            'Homogenity': metrics.homogeneity_score(y, algo.labels_),
            'Completeness': metrics.completeness_score(y, algo.labels_),
            'V-measure': metrics.v_measure_score(y, algo.labels_),
            'Silhouette': metrics.silhouette_score(X, algo.labels_)}))

    result = pd.DataFrame(data=data, columns=['ARI', 'AMI', 'Homogenity',
                                              'Completeness', 'V-measure',
                                              'Silhouette'],
                          index=['K-means', 'Affinity',
                                 'Spectral', 'Agglomerative'])

    return render_template('result_6.html', tables=[result.to_html(classes='data')], titles=result.columns.values)


#function for dendrogram building
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('distance')
        plt.ylabel('Name of Country')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            print(i)
            x = 0.5 * sum(i[1:3])
            print(x)
            y = d[1]
            print(y)
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.2g" % y, (x, y), xytext=(0, -10),
                             textcoords='offset points',
                             va='bottom', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


#create Dendrogram
@app.route('/dendro', methods=['GET', 'POST'])
def dendro():
    get_latest_data()
    # building dendrogram
    data_for_clust = df.drop(columns=['country', 'continent'], axis=1).values
    dataNorm = preprocessing.scale(data_for_clust)
    data_dist = pdist(dataNorm)

    data_linkage = linkage(data_dist, method='average')
    nCluster = len(df)

    plt.cla()
    plt.clf()
    # строим дендрограмму
    fancy_dendrogram(data_linkage, truncate_mode='lastp', p=nCluster, leaf_font_size=4.,
                     show_contracted=True, annotate_above=10, orientation = 'left', labels = df.country.values)




    label = fcluster(data_linkage, 0.5, criterion='distance')
    count_of_clusters = len(np.unique(label))
    # print(count_of_clusters)

    list = []
    list_label_product = [0, 0]
    df.loc[:, 'label'] = label
    clust_data=[]
    global list_data
    list_data=[]
    for i, country in df.groupby('label'):
        # clust_data.append("{0} {1}".format(i, country.country.values))
        list_data.append(country.country.values)
    print(list_data)
    print(list_data[1])
    print('--------------------------------------------------------------')

    plt.savefig('static/images/dendrogram.png', dpi = 1000)
    # plt.show()
    # print(clust_data)
    return render_template('result_7.html', result=list_data)


#show data from Dendrogram
@app.route('/show/<int:id>', methods=['GET', 'POST'])
def show(id):
    result = []
    print(list_data)
    for j in range(len(list_data[id-1])):
        result.append(db.session.query(Rate).filter(Rate.country_name == list_data[id-1][j]).one())

    return render_template('result_7_data.html', result=result)


#clear boofer
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


#start app
if __name__ == "__main__":
    app.run()
