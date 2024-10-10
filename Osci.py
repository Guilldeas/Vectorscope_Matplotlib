import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.interpolate import CubicSpline
from PIL import Image
from IPython.display import clear_output
import gc


###################################################################################
#                                                                                 #
#                                     TO DO                                       #
#                                                                                 #
#               Implement accurate interface to what is dispalyed                 #
#      Implement a way to select horizontal and verticla scale that doesn't       #
#                          fuck with the "hazynes"                                #
#                                                                                 #
###################################################################################


def draw_vectorscope_interface(xlims, ylims, Image_resolution=(12, 12)):
    Grid_hazyness = 0.015
    graph_opacity = 0.025
    Num_vertical_lines = 10
    Num_horizontal_lines = 10
    interface_sweeps = 200


    # Graphed text parameters
    font = 'monospace'
    plt.rcParams['font.size'] = 20  # Set default font size for all text
    plt.rcParams['font.weight'] = 'bold'  # Set default font weight (normal, bold, etc.)
    plt.rcParams['font.style'] = 'normal'  # Set default font style (normal, italic, oblique)
    #plt.rcParams['text.color'] = ''  # Set default font color
    plt.rcParams['text.usetex'] = False  # Disable LaTeX processing
    plt.rcParams['font.family'] = font  # Set to desired font

    # Simulate correct additive process and averaging effect that each scope sweep and decay time has
    image_buffers = []  # Store image arrays instead of file paths

    # Initialize the cumulative result array with zeros (set dimensions according to your plot size)
    additive_result = None
    for sweep in range(0, interface_sweeps):

        fig, ax = plt.subplots(figsize=Image_resolution, facecolor='#000000')

        # Set fixed x and y limits based on your expected data range
        ax.set_xlim(xlims)  # or whatever range your x-axis should have
        ax.set_ylim(ylims)  # Set the expected y-axis limits based on your data

        ax.set_facecolor('#000000')

        # Add random noise in the line placement to simulate gaussian blurr
        rand_x = np.random.normal(loc=0.0, scale=Grid_hazyness)
        rand_y = np.random.normal(loc=0.0, scale=Grid_hazyness)

        ############################################ Grid ############################################

        # Make a fake hacky grid line because I can't get ax.grid to do what I want.
        max_x = xlims[1]
        min_x = xlims[0]
        max_y = ylims[1]
        min_y = ylims[0]
        x_ticks_spacing = ( max_x - min_x ) / Num_vertical_lines
        y_ticks_spacing = ( max_y - min_y ) / Num_horizontal_lines

        # Text

        # Add text to simulate an oscilloscope display (adjust positions as needed)
        top_height = rand_y + max_y*1.01
        lower_height = rand_y + min_y*1.3
        line_height = 0.1

        # Draw some text that is reminiscent of the HP8566 interface

        plt.text(0 + rand_x, top_height, "0.0dBm", color=(0.04, 0.2, 0.06), alpha=graph_opacity)
        plt.text(-0.6 + rand_x, top_height, "REF", color=(0.04, 0.2, 0.06), alpha=graph_opacity)
        plt.text(1.2 + rand_x, top_height, "ATTEN 10 dB", color=(0.04, 0.2, 0.06), alpha=graph_opacity)

        plt.text(-0.6 + rand_x, lower_height, "START", color=(0.04, 0.2, 0.06), alpha=graph_opacity)
        plt.text(0.2 + rand_x, lower_height, "0 Hz", color=(0.04, 0.2, 0.06), alpha=graph_opacity)
        plt.text(8.0 + rand_x, lower_height, "RBW 1 MHz", color=(0.04, 0.2, 0.06), alpha=graph_opacity)


        for index in range(0, Num_horizontal_lines+1):

            # Add random noise in the line placement to simulate gaussian blurr
            rand_x = np.random.normal(loc=0.0, scale=Grid_hazyness)
            rand_y = np.random.normal(loc=0.0, scale=Grid_hazyness)

            # Create a vertical line defined by 2 points at the intended horizontal position that goes from minimum graph to max graph height
            x_points = [min_x, max_x]
            y_points = [min_y + index * y_ticks_spacing, min_y + index * y_ticks_spacing]
            x_points_noisy = x_points + np.array([rand_x, rand_x])
            y_points_noisy = y_points + np.array([rand_y, rand_y])

            ax.plot(x_points_noisy, y_points_noisy, linestyle='-', color=(0.04, 0.2, 0.06), linewidth = 2, alpha = graph_opacity )
        
        for index in range(0, Num_vertical_lines+1):
            # Create a vertical line defined by 2 points at the intended horizontal position that goes from minimum graph to max graph height
            x_points = [min_x + index * x_ticks_spacing, min_x + index * x_ticks_spacing]
            y_points = [min_y, max_y]
            x_points_noisy = x_points + np.array([rand_x, rand_x])
            y_points_noisy = y_points + np.array([rand_y, rand_y])

            ax.plot(x_points_noisy, y_points_noisy, linestyle='-', color=(0.04, 0.2, 0.06), linewidth = 2, alpha = graph_opacity )

        # Keep track of progress to display for user
        Sweeping_progress = (sweep / interface_sweeps)

        # Convert the figure to an image buffer (in-memory)
        canvas = FigureCanvas(fig)
        canvas.draw()

        # Convert to a NumPy array directly from the figure buffer
        img_buffer = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img_buffer = img_buffer.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Reshape to (height, width, 3)
        
        # Normalize image buffer to [0, 1]
        img_buffer = img_buffer.astype(np.float32) / 255.0

        # Initialize additive_result on first iteration
        if additive_result is None:
            additive_result = img_buffer.copy()  # Initialize with first image

        # Sum this image buffer to the accumulated sum of img buffers
        else:
            additive_result += img_buffer  # Directly add to the existing array 

        # Update user on progress
        clear_output(wait=True)
        print(f'Sweeping traces progress: {100.0}%')
        print(f'Drawing interface progress:{round(Sweeping_progress*100, 1)}%')

        plt.close('all')

    # Delete these large variables since we don't need them anymore
    del image_buffers, img_buffer, canvas
    gc.collect()
    # Return final image array to pick up from when drawing
    return additive_result







def graph_in_vectorscope (x, y, Image_resolution = (12, 12), phosphor_color = (0.04, 0.2, 0.06), beam_width = 2, beam_intensity = 0.1,
                          save_image = True, Resample_factor = 10, noise_level = 0.025, Osci_sweeps = 10, Resampling_sweeps = 10, Fade_out = True):


    Num_data_points = len(x)

    # Simulate correct additive process and averaging effect that each scope sweep and decay time has
    image_buffers = []  # Store image arrays instead of file paths

    # Initialize the cumulative result array with zeros (set dimensions according to your plot size)
    additive_result = None
    iteration = 1
    for sweep in range(0, Osci_sweeps):

        y_noisy = y + noise_level * np.random.normal(size=x.shape)

        # Cubic Spline Fit
        cubic_spline = CubicSpline(x, y_noisy)

        # Interpolating on a denser set of points
        x_dense = np.linspace(0,len(x), Num_data_points * Resample_factor)
        y_interpolated = cubic_spline(x_dense)

        # Exact derivative of spline polynomials, but calculated over x_dense
        cubic_spline_derivative = cubic_spline.derivative()
        dydx_exact = cubic_spline_derivative(x_dense)  # Evaluate derivative at x_dense

        # Compute weights based on the inverse of the derivative's magnitude
        deriv_values = np.abs(dydx_exact)

        for subsweep in range(0, Resampling_sweeps):
            fig, ax = plt.subplots(figsize=Image_resolution, facecolor='#000000')
            ax.set_facecolor('#000000')

            # Set fixed x and y limits based on your expected data range
            xlims = [x.min()*0.9, x.max()*1.1]
            ylims = [y.min()*0.9, y.max()*1.1]
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)


            weights = 1 / (deriv_values + 1e-3)  # Invert derivative values to favor low-slope regions

            # Normalize weights to sum to 1 (for use as probabilities)
            weights_normalized = weights / np.sum(weights)

            # Number of points to resample
            n_resample = Num_data_points * 10

            # Use the weights to resample non-uniformly
            x_resampled = np.random.choice(x_dense, size=n_resample, p=weights_normalized)

            # Sort the resampled x values for plotting
            x_resampled = np.sort(x_resampled)

            # Evaluate the resampled points using the cubic spline
            y_resampled = cubic_spline(x_resampled)

            # Plot the original noisy data and the resampled points
            plt.scatter(x_resampled, y_resampled, color=phosphor_color, s=beam_width, alpha=beam_intensity)

            # Convert the figure to an image buffer (in-memory)
            canvas = FigureCanvas(fig)
            canvas.draw()

            # Convert to a NumPy array directly from the figure buffer
            img_buffer = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            img_buffer = img_buffer.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Reshape to (height, width, 3)

            Sweeping_progress = iteration / (Resampling_sweeps * Osci_sweeps)
            iteration = iteration + 1

            # Choose whether to fade out traces that were drawn in the past 
            # (images) are computed inversely chronological by this logic
            if (Fade_out):
                img_buffer = img_buffer * Sweeping_progress

            # Normalize image buffer to [0, 1]
            img_buffer = img_buffer.astype(np.float32) / 255.0

            # Initialize additive_result on first iteration
            if additive_result is None:
                additive_result = img_buffer.copy()  # Initialize with first image

            # Sum this image buffer to the accumulated sum of img buffers
            else:
                additive_result += img_buffer  # Directly add to the existing array 

            # Update user on progress
            clear_output(wait=True)
            print(f'Sweeping traces progress: {round(Sweeping_progress*100, 1)}%')

            # Close before displaying graphs on output cell
            plt.close('all')

    # Delete these large variables since we don't need them anymore
    del image_buffers, img_buffer


    # Add all image arrays in a way that is not memory intensive
    #batch_size = math.floor(len(image_buffers) /  10)
    #additive_result, _ = add_matrices_in_batches(Matrices_List=image_buffers, batch_size=batch_size, Troubleshooting = False)

    # Step 3: Compute interface image separately and then add it to graph
    additive_result = np.sum([additive_result, draw_vectorscope_interface(xlims=xlims, ylims=ylims, Image_resolution=Image_resolution)], axis=0)

    # Step 4: Clip values to stay within the valid RGB range (0 to 1)
    additive_result = np.clip(additive_result, 0, 1)

    # Step 5: Convert the result back to an image
    final_image = Image.fromarray((additive_result * 255).astype(np.uint8))  # Convert back to 8-bit format
    del additive_result

    # Show or save the final image
    final_image.show()

    if (save_image):
        final_image.save('Python_Output/Final_graph_SweepNum_' + str(sweep) 
                        + '_SubSweepNum_' + str(subsweep)  
                        + '_ResampleFactor_' + str(Resample_factor) 
                        + '_Num_data_points_' + str(Num_data_points) 
                        + '_Fadeout_' + str(Fade_out) 
                        + '.png')
    final_image.close()

    clear_output(wait=True)
    print(f'Sweeping traces progress: {100.0}%')
    print(f'Drawing interface progress:{100.0}%')

    del canvas, fig, deriv_values, dydx_exact, weights, weights_normalized, x, x_resampled
    del y_resampled, y_interpolated, y_noisy, final_image, cubic_spline, cubic_spline_derivative, x_dense




# Gaussian function: mean = 0, standard deviation = 1

x = np.linspace(0, 10, 100)
mu = 0  # Mean
sigma = 1  # Standard deviation
gaussian = (10 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - 5 - mu)**2 / (2 * sigma**2))

graph_in_vectorscope (x=x, y=gaussian, Image_resolution = (12, 12), phosphor_color = (0.04, 0.2, 0.06), beam_width = 2, beam_intensity = 0.1,
                          save_image = True, Resample_factor = 10, noise_level = 0.025, Osci_sweeps = 5, Resampling_sweeps = 10, Fade_out = True)

# Release memory at the end
gc.collect() 