from visualize_helpers import *
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Visualization options")

    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Run t-SNE visualization"
    )

    parser.add_argument(
        "--sample_num",
        type=int,
        default=None,
        help="Sample num to visualize"
    )

    parser.add_argument(
        "--input_dist",
        action="store_true",
        help="Visualize input distribution"
    )

    return parser.parse_args()

def main():
    
    PERPLEXITY = 30
    RANGE = 2000
    
    args = parse_args()
    
    ##################################################
    #### t-SNE plots of image modalities ####
    ##################################################
    if args.tsne or args.sample_num is not None:
        
        # Generate CLIP embeddings for the grayscale images
        save_path_grayscale_embeds = '/orcd/data/faez/001/annie/mmai/data/grayscale_clip_embds.npy'
        use_existing = False
        if os.path.exists(save_path_grayscale_embeds):
            user_input = input(
                    f"Image embeddings file already exists at {save_path_grayscale_embeds}.\n"
                    "Do you want to regenerate? (y/n): "
                )
            if user_input.lower() != "y":
                use_existing = True
                
        if not use_existing:      
            grayscale_image_inputs = [f"/orcd/data/faez/001/annie/mmai/data/images/{i}.png" for i in range(RANGE)]
            # test_images = grayscale_image_inputs[:100]
            gray_scale_clip_embds = get_clip_embeddings(grayscale_image_inputs, batch_size=1000)
            np.save(save_path_grayscale_embeds, gray_scale_clip_embds)
            print(f"Saved embeddings to {save_path_grayscale_embeds}")
        else:
            gray_scale_clip_embds = np.load(save_path_grayscale_embeds)

        labels = np.arange(len(gray_scale_clip_embds))

        visualize_data_distribution(gray_scale_clip_embds, num_components = 2, perplexity = PERPLEXITY, num_iterations=1000, savename = "visualizations/grayscale_images_tsne.html", labels=labels, hover_text=labels, color_by_label = False)

        # Generate CLIP embeddings for the synthetic images
        save_path_synthetic_embeds = "/orcd/data/faez/001/annie/mmai/data/synthetic_clip_embds.npy"
        use_existing = False
        if os.path.exists(save_path_synthetic_embeds):
            user_input = input(
                f"Synthetic image embeddings file already exists at {save_path_synthetic_embeds}.\n"
                "Do you want to regenerate? (y/n): "
            )
            if user_input.lower() != "y":
                use_existing = True

        if not use_existing:
            synthetic_image_inputs = [f"/orcd/data/faez/001/annie/mmai/data/syn_real/{i}_gemini.png" for i in range(RANGE)]
            synthetic_clip_embds = get_clip_embeddings(synthetic_image_inputs, batch_size=1000)
            np.save(save_path_synthetic_embeds, synthetic_clip_embds)
            print(f"Saved embeddings to {save_path_synthetic_embeds}")

        else:
            synthetic_clip_embds = np.load(save_path_synthetic_embeds)

        labels = np.arange(len(synthetic_clip_embds))

        visualize_data_distribution(synthetic_clip_embds, num_components = 2, perplexity = PERPLEXITY, num_iterations=1000, savename = "visualizations/synthetic_images_tsne.html", labels=labels, hover_text=labels, color_by_label = False)
            
    
    ##################################################
    #### Visualize samples ####
    ##################################################
    if args.sample_num is not None:
        visualize_grayscale_image(args.sample_num)
        visualize_photorealistic_image(args.sample_num)
        visualize_point_cloud(args.sample_num)
        print("*" * 50)
        visualize_code(args.sample_num)
        visualize_description(args.sample_num)
        print("*" * 50)
        visualize_data_distribution(gray_scale_clip_embds, num_components = 2, perplexity = PERPLEXITY, num_iterations=1000, savename = f"visualizations/grayscale_images_tsne.html", labels=labels, hover_text=labels, color_by_label = False, highlight_indices=args.sample_num)
        visualize_data_distribution(synthetic_clip_embds, num_components = 2, perplexity = PERPLEXITY, num_iterations=1000, savename = f"visualizations/synthetic_images_tsne.html", labels=labels, hover_text=labels, color_by_label = False, highlight_indices=args.sample_num)

    ##################################################
    #### Visualize input distribution ####
    ##################################################
    if args.input_dist:
        get_command_counts(RANGE)

if __name__ == "__main__":
    main()