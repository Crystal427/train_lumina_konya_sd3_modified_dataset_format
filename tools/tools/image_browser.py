import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import gradio as gr
from PIL import Image


# ----------------------------
# Constants
# ----------------------------
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ----------------------------
# Utility helpers
# ----------------------------
def list_artist_dirs(root: Path) -> List[Path]:
    """List all artist directories in the root."""
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def splitext_lower(name: str) -> tuple[str, str]:
    """Split filename and return lowercase extension."""
    base, ext = os.path.splitext(name)
    return base, ext.lower()


def find_all_images(artist_dir: Path) -> List[Path]:
    """Find all images in an artist directory (all subdirectories)."""
    results: List[Path] = []
    
    for root, dirs, files in os.walk(artist_dir):
        for file in files:
            base, ext = splitext_lower(file)
            if ext in VALID_IMAGE_EXTS:
                results.append(Path(root) / file)
    
    return sorted(results)


def open_folder_in_explorer(folder_path: Path) -> None:
    """Open folder in system file explorer."""
    try:
        if sys.platform == "win32":
            os.startfile(str(folder_path))
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", str(folder_path)])
        else:  # Linux
            subprocess.run(["xdg-open", str(folder_path)])
    except Exception as e:
        print(f"Failed to open folder: {e}", file=sys.stderr)


# ----------------------------
# Image Browser State
# ----------------------------
class ImageBrowserState:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.artists = list_artist_dirs(root_path)
        self.current_artist_idx = 0
        self.current_images: List[Path] = []
        self.selected_indices: List[int] = []  # Track selected images in gallery
        
        if self.artists:
            self._load_artist(0)
    
    def _load_artist(self, artist_idx: int) -> None:
        """Load images for a specific artist."""
        if 0 <= artist_idx < len(self.artists):
            self.current_artist_idx = artist_idx
            self.current_images = find_all_images(self.artists[artist_idx])
            self.selected_indices = []
    
    def get_current_artist_name(self) -> str:
        """Get current artist name."""
        if not self.artists:
            return "No artists found"
        return self.artists[self.current_artist_idx].name
    
    def get_info_text(self) -> str:
        """Get information text for current state."""
        if not self.artists:
            return "No artists found in directory"
        
        artist_name = self.get_current_artist_name()
        total_artists = len(self.artists)
        total_images = len(self.current_images)
        
        if total_images == 0:
            return f"Artist: {artist_name} ({self.current_artist_idx + 1}/{total_artists})\nNo images found"
        
        selected_info = ""
        if self.selected_indices:
            selected_info = f"\nSelected: {len(self.selected_indices)} image(s)"
        
        return (
            f"Artist: {artist_name} ({self.current_artist_idx + 1}/{total_artists})\n"
            f"Total Images: {total_images}"
            f"{selected_info}"
        )
    
    def get_gallery_images(self) -> List[str]:
        """Get list of image paths as strings for gallery."""
        return [str(img_path) for img_path in self.current_images]
    
    def next_artist(self) -> tuple[List[str], str]:
        """Go to next artist."""
        if not self.artists:
            return [], self.get_info_text()
        
        next_idx = (self.current_artist_idx + 1) % len(self.artists)
        self._load_artist(next_idx)
        return self.get_gallery_images(), self.get_info_text()
    
    def prev_artist(self) -> tuple[List[str], str]:
        """Go to previous artist."""
        if not self.artists:
            return [], self.get_info_text()
        
        prev_idx = (self.current_artist_idx - 1) % len(self.artists)
        self._load_artist(prev_idx)
        return self.get_gallery_images(), self.get_info_text()
    
    def update_selection(self, evt: gr.SelectData) -> str:
        """Update selected image based on gallery selection."""
        if evt.selected:
            if evt.index not in self.selected_indices:
                self.selected_indices.append(evt.index)
        else:
            if evt.index in self.selected_indices:
                self.selected_indices.remove(evt.index)
        
        return self.get_info_text()
    
    def delete_selected_images(self) -> tuple[List[str], str, str]:
        """Delete all selected images."""
        if not self.selected_indices:
            return self.get_gallery_images(), self.get_info_text(), "No images selected"
        
        # Sort indices in reverse order to delete from end to start
        sorted_indices = sorted(self.selected_indices, reverse=True)
        deleted_count = 0
        
        for idx in sorted_indices:
            if 0 <= idx < len(self.current_images):
                img_path = self.current_images[idx]
                try:
                    img_path.unlink()
                    print(f"Deleted: {img_path}")
                    self.current_images.pop(idx)
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {img_path}: {e}", file=sys.stderr)
        
        self.selected_indices = []
        status_msg = f"Deleted {deleted_count} image(s)"
        
        return self.get_gallery_images(), self.get_info_text(), status_msg
    
    def open_current_artist_folder(self) -> str:
        """Open current artist folder in file explorer."""
        if not self.artists:
            return "No artist folder available"
        
        artist_path = self.artists[self.current_artist_idx]
        open_folder_in_explorer(artist_path)
        return f"Opened folder: {artist_path}"


# ----------------------------
# Gradio UI
# ----------------------------
def create_ui(root_path: Path):
    """Create Gradio interface for image browsing."""
    state = ImageBrowserState(root_path)
    
    with gr.Blocks(title="Image Gallery Browser", css="""
        .gallery-container {min-height: 600px;}
    """) as demo:
        gr.Markdown("# 图片画廊浏览器")
        gr.Markdown("点击图片选择/取消选择，支持多选，然后点击删除按钮")
        
        with gr.Row():
            info_text = gr.Textbox(
                label="信息",
                value=state.get_info_text(),
                lines=3,
                interactive=False,
            )
        
        with gr.Row():
            gallery = gr.Gallery(
                label="图片画廊",
                value=state.get_gallery_images(),
                columns=12,
                rows=4,
                height=600,
                object_fit="contain",
                allow_preview=True,
                selected_index=None,
                elem_classes=["gallery-container"],
            )
        
        with gr.Row():
            prev_artist_btn = gr.Button("⏮ 上一个艺术家", size="sm")
            next_artist_btn = gr.Button("下一个艺术家 ⏭", size="sm")
            delete_btn = gr.Button("🗑 删除选中图片", variant="stop", size="lg")
            open_folder_btn = gr.Button("📁 打开文件夹", variant="primary", size="sm")
        
        with gr.Row():
            status_text = gr.Textbox(label="状态", value="就绪", interactive=False)
        
        # Button callbacks
        def on_prev_artist():
            images, info = state.prev_artist()
            return images, info, f"切换到艺术家: {state.get_current_artist_name()}"
        
        def on_next_artist():
            images, info = state.next_artist()
            return images, info, f"切换到艺术家: {state.get_current_artist_name()}"
        
        def on_delete():
            images, info, status = state.delete_selected_images()
            return images, info, status
        
        def on_open_folder():
            return state.open_current_artist_folder()
        
        def on_select(evt: gr.SelectData):
            # Track selection
            info = state.update_selection(evt)
            return info
        
        # Wire up buttons
        prev_artist_btn.click(
            on_prev_artist,
            outputs=[gallery, info_text, status_text],
        )
        
        next_artist_btn.click(
            on_next_artist,
            outputs=[gallery, info_text, status_text],
        )
        
        delete_btn.click(
            on_delete,
            outputs=[gallery, info_text, status_text],
        )
        
        # Gallery selection tracking
        gallery.select(
            on_select,
            outputs=[info_text],
        )
        
        open_folder_btn.click(
            on_open_folder,
            outputs=[status_text],
        )
        
        # Add keyboard shortcuts info
        gr.Markdown("""
        ### 使用说明:
        - **点击图片**: 选择/取消选择图片（支持多选）
        - **删除选中图片**: 删除所有选中的图片
        - **切换艺术家**: 查看不同艺术家文件夹的图片
        - **打开文件夹**: 在文件管理器中打开当前艺术家文件夹
        """)
        
        # Keyboard shortcuts via JavaScript
        demo.load(
            None,
            None,
            None,
            js="""
            function() {
                if (!window.galleryKeyboardHandlerAdded) {
                    window.galleryKeyboardHandlerAdded = true;
                    document.addEventListener('keydown', function(e) {
                        // Get all buttons
                        const buttons = document.querySelectorAll('button');
                        let deleteBtn = null;
                        let prevArtistBtn = null;
                        let nextArtistBtn = null;
                        
                        // Find buttons by text content
                        buttons.forEach(btn => {
                            const text = btn.textContent.trim();
                            if (text.includes('删除选中图片')) deleteBtn = btn;
                            if (text.includes('上一个艺术家')) prevArtistBtn = btn;
                            if (text.includes('下一个艺术家')) nextArtistBtn = btn;
                        });
                        
                        // Handle key press
                        if (e.key === 'Delete' && deleteBtn) {
                            e.preventDefault();
                            deleteBtn.click();
                        } else if (e.key === 'ArrowLeft' && e.ctrlKey && prevArtistBtn) {
                            e.preventDefault();
                            prevArtistBtn.click();
                        } else if (e.key === 'ArrowRight' && e.ctrlKey && nextArtistBtn) {
                            e.preventDefault();
                            nextArtistBtn.click();
                        }
                    });
                }
            }
            """,
        )
    
    return demo


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactive image gallery browser with multi-select and batch delete"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing artist folders with images",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the web interface (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    
    args = parser.parse_args(argv)
    root_path = Path(args.root)
    
    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_path}", file=sys.stderr)
        return 1
    
    print(f"Starting image gallery browser for: {root_path}")
    print(f"Server will run on port: {args.port}")
    
    demo = create_ui(root_path)
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        allowed_paths=[str(root_path.resolve())],
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
