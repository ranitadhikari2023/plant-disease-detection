/**
 * web/src/components/Dropzone.tsx
 * Specimen drag-and-drop zone and file/camera selectors
 */

import React, { useRef } from 'react';
import { useDragCounter } from '../hooks/useDragCounter';
import styles from './Dropzone.module.css';

interface DropzoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export const Dropzone: React.FC<DropzoneProps> = ({ onFileSelect, disabled = false }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const { isDragging, onDragEnter, onDragOver, onDragLeave, onDrop } = useDragCounter();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
  };

  const handleContainerClick = (e: React.MouseEvent) => {
    if (disabled) return;
    if ((e.target as HTMLElement).closest('button')) return;
    fileInputRef.current?.click();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (disabled) return;
    if (e.key === 'Enter' || e.key === ' ') {
      if (!(e.target as HTMLElement).closest('button')) {
        e.preventDefault();
        fileInputRef.current?.click();
      }
    }
  };

  return (
    <div
      className={`${styles.dropzone} ${isDragging ? styles.dragover : ''} ${
        disabled ? styles.disabled : ''
      }`}
      onDragEnter={disabled ? undefined : onDragEnter}
      onDragOver={disabled ? undefined : onDragOver}
      onDragLeave={disabled ? undefined : onDragLeave}
      onDrop={
        disabled
          ? undefined
          : (e) =>
              onDrop(e, (files) => {
                if (files.length > 0) onFileSelect(files[0]);
              })
      }
      onClick={handleContainerClick}
      onKeyDown={handleKeyDown}
      tabIndex={disabled ? -1 : 0}
      role="region"
      aria-label="Leaf specimen photograph drop area"
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
        style={{ display: 'none' }}
        onChange={handleFileChange}
        disabled={disabled}
      />
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        style={{ display: 'none' }}
        onChange={handleFileChange}
        disabled={disabled}
      />

      <div className={styles.iconCircle}>
        <svg
          width="32"
          height="32"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.75"
          aria-hidden="true"
        >
          <path d="M12 2a10 10 0 0 1 10 10c0 5.523-4.477 10-10 10S2 17.523 2 12 6.477 2 12 2z" />
          <path d="M12 8v8" />
          <path d="m8 12 4-4 4 4" />
        </svg>
      </div>

      <div className={styles.content}>
        <h3 className={styles.heading}>Drop your leaf photograph here</h3>
        <p className={styles.subheading}>
          Drag & drop a crisp foliar image, or choose an input source below
        </p>

        <div className={styles.buttonGroup}>
          <button
            type="button"
            className="btn btn-primary"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
            disabled={disabled}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              aria-hidden="true"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            Browse Files
          </button>

          <button
            type="button"
            className="btn btn-secondary"
            onClick={(e) => {
              e.stopPropagation();
              cameraInputRef.current?.click();
            }}
            disabled={disabled}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              aria-hidden="true"
            >
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
              <circle cx="12" cy="13" r="4" />
            </svg>
            Take Photo
          </button>
        </div>

        <p className={styles.formats}>Supported: JPEG, PNG, WebP (Max 10 MB)</p>
      </div>
    </div>
  );
};
