import { useState, useRef } from 'react'
import { Crop, Loader2, Check, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import Cropper, { type ReactCropperElement } from "react-cropper"
import "cropperjs/dist/cropper.css"
import { createPortal } from 'react-dom'

interface ScreenshotButtonProps {
    onScreenshot: (file: File) => void
    disabled?: boolean
}

export function ScreenshotButton({ onScreenshot, disabled }: ScreenshotButtonProps) {
    const [isCapturing, setIsCapturing] = useState(false)
    const [screenImage, setScreenImage] = useState<string | null>(null)
    const cropperRef = useRef<ReactCropperElement>(null)

    const handleCapture = async () => {
        try {
            setIsCapturing(true)
            
            // 1. Get screen stream
            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: { cursor: "always" } as any,
                audio: false
            })

            const video = document.createElement('video')
            video.srcObject = stream
            
            await new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    video.play()
                    resolve(null)
                }
            })

            // Wait a bit for the video to actually render the frame
            await new Promise(r => setTimeout(r, 300))

            // 2. Draw to canvas
            const canvas = document.createElement('canvas')
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
            const ctx = canvas.getContext('2d')
            
            if (!ctx) throw new Error("Could not get canvas context")
            
            ctx.drawImage(video, 0, 0)
            
            // Stop stream tracks immediately after capture
            stream.getTracks().forEach(track => track.stop())

            // 3. Get data URL for Cropper
            const dataUrl = canvas.toDataURL('image/png')
            setScreenImage(dataUrl)

        } catch (error) {
            console.error("Screenshot error:", error)
        } finally {
            setIsCapturing(false)
        }
    }

    const handleCropConfirm = () => {
        const cropper = cropperRef.current?.cropper;
        if (cropper) {
            cropper.getCroppedCanvas().toBlob((blob) => {
                if (blob) {
                    const file = new File([blob], `screenshot-${Date.now()}.png`, { type: 'image/png' })
                    onScreenshot(file)
                }
                setScreenImage(null) // Close modal
            }, 'image/png')
        }
    }

    const handleCancel = () => {
        setScreenImage(null)
    }

    return (
        <>
            <Button 
                type="button" 
                variant="ghost" 
                size="icon"
                onClick={handleCapture}
                disabled={disabled || isCapturing}
                title="屏幕截图"
            >
                {isCapturing ? <Loader2 className="w-5 h-5 animate-spin text-blue-500" /> : <Crop className="w-5 h-5 text-gray-500" />}
            </Button>

            {screenImage && createPortal(
                <div className="fixed inset-0 z-[9999] bg-black/80 flex flex-col items-center justify-center p-4 animate-in fade-in duration-200">
                    <div className="relative w-full h-full max-h-[85vh] bg-black rounded-lg overflow-hidden shadow-2xl border border-gray-700">
                        <Cropper
                            src={screenImage}
                            style={{ height: "100%", width: "100%" }}
                            initialAspectRatio={NaN}
                            guides={false} // No grid lines
                            viewMode={1}
                            dragMode="crop" // Draw crop box
                            autoCrop={false} // Start with no selection
                            movable={false} // Disable image moving
                            zoomable={false} // Disable image zooming
                            scalable={false}
                            ref={cropperRef}
                            background={false}
                            responsive={true}
                            className="screenshot-cropper"
                        />
                        
                        {/* Control Bar */}
                        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-4 bg-white/10 backdrop-blur-md p-3 rounded-full border border-white/20 shadow-xl z-50">
                            <Button 
                                onClick={handleCancel} 
                                variant="secondary" 
                                size="icon" 
                                className="rounded-full h-12 w-12 bg-white/80 hover:bg-white text-gray-800"
                                title="取消"
                            >
                                <X className="w-6 h-6" />
                            </Button>
                            <Button 
                                onClick={handleCropConfirm} 
                                className="rounded-full h-12 w-12 bg-green-600 hover:bg-green-500 text-white border-2 border-transparent hover:border-green-300"
                                size="icon"
                                title="确认截图"
                            >
                                <Check className="w-6 h-6" />
                            </Button>
                        </div>

                        {/* Instructions */}
                        <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-black/50 text-white px-4 py-1 rounded-full text-sm backdrop-blur-sm pointer-events-none">
                            拖拽框选区域 • 滚轮缩放
                        </div>
                    </div>
                </div>,
                document.body
            )}
        </>
    )
}
