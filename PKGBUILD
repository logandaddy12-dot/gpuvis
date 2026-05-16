# Maintainer: logandaddy12-dot
pkgname=gpuvis
pkgver=1.0
pkgrel=1
pkgdesc="GPU usage visualizer with particles"
arch=('x86_64')
url="https://github.com/logandaddy12-dot/gpuvis"
license=('MIT')
depends=('glfw-x11' 'glew' 'nvidia-utils')
makedepends=('gcc')
source=("gpuvis.cpp::https://raw.githubusercontent.com/logandaddy12-dot/gpuvis/refs/heads/main/gpuvis.cpp")
sha256sums=('SKIP')

build(){
    cd "$srcdir"
    g++ -O2 -o gpuvis gpuvis.cpp $(pkg-config --libs --cflags glfw3 glew) -lGL -std=c++17 -lm -lpthread
}

package(){
    install -Dm755 "$srcdir/gpuvis" "$pkgdir/usr/bin/gpuvis"
}
