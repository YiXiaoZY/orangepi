"""注册人脸到库: 把照片放到 data/face_db/姓名/ 下，然后运行本脚本"""
import sys

from security.face_database import FaceDatabase, enroll_from_folders
from security.face_embedder import FaceEmbedder


def main():
    print("扫描 data/face_db/ 注册人脸...")
    with FaceEmbedder() as embedder:
        db = enroll_from_folders(embedder)
    if db.count() == 0:
        print("\n未注册任何人。请创建目录并放入照片，例如:")
        print("  data/face_db/张三/photo1.jpg")
        print("  data/face_db/李四/photo2.jpg")
        sys.exit(1)
    print(f"当前库内人员: {db.list_persons()}")


if __name__ == "__main__":
    main()
