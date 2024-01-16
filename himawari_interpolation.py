from __future__ import division
import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree as KDTree
import get_thredds_file




# Davies
TARGET_LATITUDE = -18.83162
TARGET_LONGITUDE = 147.6345



""" invdisttree.py: inverse-distance-weighted interpolation using KDTree
    fast, solid, local
"""

# http://docs.scipy.org/doc/scipy/reference/spatial.html
# https://en.wikipedia.org/wiki/Inverse_distance_weighting
# https://en.wikipedia.org/wiki/K-d_tree

# ...............................................................................
class Invdisttree:
    """ inverse-distance-weighted interpolation using KDTree:
invdisttree = Invdisttree( X, z )  -- data points, values
interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
    interpolates z from the 3 points nearest each query point q;
    For example, interpol[ a query point q ]
    finds the 3 data points nearest q, at distances d1 d2 d3
    and returns the IDW average of the values z1 z2 z3
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

    q may be one point, or a batch of points.
    eps: approximate nearest, dist <= (1 + eps) * true nearest
    p: use 1 / distance**p
    weights: optional multipliers for 1 / distance**p, of the same shape as q
    stat: accumulate wsum, wn for average weights

How many nearest neighbors should one take ?
a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
    |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
    I find that runtimes don't increase much at all with nnear -- ymmv.

p=1, p=2 ?
    p=2 weights nearer points more, farther points less.
    In 2d, the circles around query points have areas ~ distance**2,
    so p=2 is inverse-area weighting. For example,
        (z1/area1 + z2/area2 + z3/area3)
        / (1/area1 + 1/area2 + 1/area3)
        = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
    Similarly, in 3d, p=3 is inverse-volume weighting.

Scaling:
    if different X coordinates measure different things, Euclidean distance
    can be way off.  For example, if X0 is in the range 0 to 1
    but X1 0 to 1000, the X1 distances will swamp X0;
    rescale the data, i.e. make X0.std() ~= X1.std() .

A nice property of IDW is that it's scale-free around query points:
if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
the IDW average
    (z1/d1 + z2/d2 + z3/d3)
    / (1/d1 + 1/d2 + 1/d3)
is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
is exceedingly sensitive to distance and to h.

    """


    def __init__(self, X, z, leafsize=10, stat=0):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree(X, leafsize=leafsize)  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None;

    def __call__(self, q, nnear=6, eps=0, p=1, weights=None):
        # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query(q, k=nnear, eps=eps)
        interpol = np.zeros((len(self.distances),) + np.shape(self.z[0]))
        jinterpol = 0
        for dist, ix in zip(self.distances, self.ix):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist ** p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot(w, self.z[ix])
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1 else interpol[0]


# ...............................................................................
if __name__ == "__main__":
    import sys

    # get thredds url
    file_urls = get_thredds_file.get_thredds_file_urls()
    file_interpolation_output = []
    # file_url = 'https://dapds00.nci.org.au/thredds/dodsC/rv74/satellite-products/arc/der/himawari-ahi/solar/p1d/latest/2020/01/01/IDE02326.202001010000.nc'

    for file_url in file_urls:
        # open netcdf dataset
        ds = xr.open_dataset(file_url)

        # Sort out data

        # Filter the data by coordinate to reduce interpolation load
        FILTER_DEGREES = 2
        filtered_ds = ds.where(
            (ds['latitude'] > TARGET_LATITUDE - FILTER_DEGREES) &
            (ds['latitude'] < TARGET_LATITUDE + FILTER_DEGREES) &
            (ds['longitude'] > TARGET_LONGITUDE - FILTER_DEGREES) &
            (ds['longitude'] < TARGET_LONGITUDE + FILTER_DEGREES),
            drop=True)
        data = filtered_ds['daily_integral_of_surface_global_irradiance'][0].values
        # data = filtered_ds['hourly_integral_of_surface_global_irradiance'][0].values

        # Convert coordinates into useable format
        x = filtered_ds['longitude'].values
        y = filtered_ds['latitude'].values

        coords_i = []
        coords_j = []
        for i in range(len(y)):
            for j in range(len(x)):
                coords_j.append([y[i], x[j]])
            coords_i.append(coords_j)
        converted_coordinates = coords_i[0]

        # Convert data to useable format
        data_convert_list = []
        for i in data:
            for j in i:
                data_convert_list.append(j)
        converted_data = np.array(data_convert_list)

        # Interpolate with griddata through nearest 4 values
        # Flatten the coordinates for interpolation
        x_grid, y_grid = np.meshgrid(x, y)
        points = np.column_stack((x_grid.flatten(), y_grid.flatten()))

        # Flatten the values for interpolation
        values = data.flatten()

        # Interpolate at the target coordinates
        interpolated_value = griddata(points, values, (TARGET_LONGITUDE, TARGET_LATITUDE), method='linear')

        # print(f'Interpolated Value at ({target_x}, {target_y}): {interpolated_value[0]}')
        print("Basic Interpolation: ", interpolated_value)

        N = len(converted_coordinates)
        Ndim = 2
        Nask = N  # N Nask 1e5: 24 sec 2d, 27 sec 3d on mac g4 ppc
        Nnear = 8  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
        leafsize = 10
        eps = 0.1  # approximate nearest, dist <= (1 + eps) * true nearest
        p = 1  # weights ~ 1 / distance**p
        cycle = 0.25
        seed = 1

        "\n".join(sys.argv[1:])  # python this.py N= ...
        np.random.seed(seed)
        np.set_printoptions(3, threshold=100, suppress=True)  # .3f

        print("\nInvdisttree:  N %d  Ndim %d  Nask %d  Nnear %d  leafsize %d  eps %.2g  p %.2g" % (
            N, Ndim, Nask, Nnear, leafsize, eps, p))

        # list of coorinates
        known_coordinates = converted_coordinates

        # List of data values
        known_data_points = converted_data

        # list of target coordinates
        target_location = [TARGET_LATITUDE, TARGET_LONGITUDE]


        # def terrain(x):
        #     """ ~ rolling hills """
        #     return np.sin((2 * np.pi / cycle) * np.mean(x, axis=-1))


        # ...............................................................................
        invdisttree = Invdisttree(known_coordinates, known_data_points, leafsize=leafsize, stat=1)
        interpol = invdisttree(target_location, nnear=Nnear, eps=eps, p=p)
        #     invdisttree = Invdisttree( known_coordinates, known_data_points )  -- data points, values
        #     interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
        #     interpolates known_data_points from the 3 points nearest each query point q;
        #     For example, interpol[ a query point q ]
        #     finds the 3 data points nearest q, at distances d1 d2 d3
        #     and returns the IDW average of the values z1 z2 z3
        #         (z1/d1 + z2/d2 + z3/d3)
        #         / (1/d1 + 1/d2 + 1/d3)
        #         = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

        #     q may be one point, or a batch of points.
        #     eps: approximate nearest, dist <= (1 + eps) * true nearest
        #     p: use 1 / distance**p
        #     weights: optional multipliers for 1 / distance**p, of the same shape as q
        #     stat: accumulate wsum, wn for average weights

        print("average distances to nearest points: %s" % \
              np.mean(invdisttree.distances, axis=0))
        weights = (invdisttree.wsum / invdisttree.wn)
        print("average weights: %s" % weights)
        #         # see Wikipedia Zipf's law
        #     err = np.abs( terrain(target_location) - interpol )
        #     print(err)
        #     print ("average |terrain() - interpolated|: %.2g" % np.mean(err))
        print("Target Location: ", target_location)
        print("Interpolated Value: ", interpol)

        file_interpolation_output.append((file_url, TARGET_LATITUDE, TARGET_LONGITUDE, interpol, N, Ndim, Nask, Nnear, leafsize, eps, p))
    df = pd.DataFrame(file_interpolation_output)

    # saving the dataframe
    df.to_csv('test_davies.csv')
    #     interpol