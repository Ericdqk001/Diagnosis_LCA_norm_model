from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

images_path = Path("src", "discover", "images")

modalities = [
    "CT",
    "CV",
    "FC",
    "SA",
]

# Set up the figure and GridSpec for subplots
fig = plt.figure(figsize=(10, 20))
gs = gridspec.GridSpec(
    4, 1, hspace=0
)  # Set hspace to 0 to close the gap between subplots

# Plot each modality in a separate subplot
for i, mod in enumerate(modalities):
    latent_deviation_group_diffs = Path(
        images_path, f"latent_deviation_difference_{mod}.png"
    )

    # Read the image
    img = mpimg.imread(latent_deviation_group_diffs)

    # Create a subplot in the GridSpec
    ax = fig.add_subplot(gs[i, 0])
    ax.imshow(img)
    ax.axis("off")  # Hide the axis

# Adjust subplot parameters to close the gaps
plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)

# Display the plot
plt.show()
