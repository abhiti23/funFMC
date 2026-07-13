import numpy as np
import nibabel as nib
import nilearn.plotting as plotting
from nilearn import datasets, surface
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# read in the estimated loading matrix
DIR = "../results"
A = np.loadtxt(f"{DIR}/est_A.csv", delimiter=",")
A = A[1:]  # first line is just a header
p, K = A.shape
print("Shape:", p, K)

# thresholding the matrix
THRESHOLD = 0.10
A[np.abs(A) <= THRESHOLD] = 0.0
print(f"Entries zeroed out: {np.sum(A == 0)} of {p*K} ...")

# Load Schaefer 200 atlas
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200)
atlas_img = nib.load(schaefer.maps)

# Load fsaverage5 surface (lower resolution, faster; use fsaverage for publication)
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

def membership_to_nifti(atlas_img, membership_vec):
    """Map a (p,) vector of parcel values onto a 3D NIfTI volume."""
    atlas_data = atlas_img.get_fdata()
    out = np.zeros_like(atlas_data, dtype=float)
    for parcel_idx, value in enumerate(membership_vec):
        parcel_label = parcel_idx + 1  # Schaefer labels start at 1
        out[atlas_data == parcel_label] = value
    return nib.Nifti1Image(out, atlas_img.affine)

def vol_to_surf(nifti_img, mesh):
    """Project a volumetric NIfTI image onto a surface mesh."""
    return surface.vol_to_surf(nifti_img, mesh)

# For each cluster we show 4 views: left lateral, left medial, right lateral, right medial
# This gives a complete picture of the cortex without the overlap problem of glass brains
fig = plt.figure(figsize=(16, 12), facecolor="white")
gs = gridspec.GridSpec(K, 4, figure=fig, hspace=0.05, wspace=0.02)

vmin, vmax = -1, 1

# Add column labels at figure level before plotting (avoids 3D axis text issues)
col_labels = ["Left lateral", "Left medial", "Right lateral", "Right medial"]
col_x_positions = [0.20, 0.40, 0.60, 0.79]  # approximate figure-level x
# centres
for x, label in zip(col_x_positions, col_labels):
    fig.text(x, 0.97, label, ha='center', fontsize=20, color='gray')

for k in range(K):
    membership_map = membership_to_nifti(atlas_img, A[:, k])

    # Project volume onto left and right surfaces
    texture_left  = vol_to_surf(membership_map, fsaverage.pial_left)
    texture_right = vol_to_surf(membership_map, fsaverage.pial_right)

    views = [
        (texture_left,  fsaverage.pial_left,  fsaverage.sulc_left,  "lateral", f"Cluster {k+1}"),
        (texture_left,  fsaverage.pial_left,  fsaverage.sulc_left,  "medial",  ""),
        (texture_right, fsaverage.pial_right, fsaverage.sulc_right, "lateral", ""),
        (texture_right, fsaverage.pial_right, fsaverage.sulc_right, "medial",  ""),
    ]

    for col, (texture, mesh, bg_map, view, title) in enumerate(views):
        ax = fig.add_subplot(gs[k, col], projection='3d')
        plotting.plot_surf_stat_map(
            mesh,
            texture,
            hemi='left' if col < 2 else 'right',
            view=view,
            bg_map=bg_map,           # sulcal depth for anatomical context
            bg_on_data=True,
            cmap='RdBu_r',
            vmax=vmax,
            colorbar=False,
            axes=ax,
            figure=fig,
            #darkness=0.5,            # how much sulcal shading shows through
        )
        if title:
            ax.set_title(title, fontsize=13, fontweight='bold', pad=2)

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Loading value", fontsize=11)

plt.savefig(f"{DIR}/all_clusters_surface.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved surface plot.")
