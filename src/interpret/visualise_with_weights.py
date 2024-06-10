from nilearn import datasets, surface

fsaverage = datasets.fetch_surf_fsaverage()
motor_images = datasets.fetch_neurovault_motor_task()
mesh = surface.load_surf_mesh(fsaverage.pial_right)
map = surface.vol_to_surf(motor_images.images[0], mesh)
