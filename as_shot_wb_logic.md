### darktable “as shot” 白平衡处理逻辑详解

darktable 在处理 RAW 图像的 “as shot” (机内设置) 白平衡时，采用的是一套严谨的多层递退策略（fallback strategy），确保在任何情况下都能找到一组可用的白平衡系数。其核心思想是优先使用最精确、来源最可靠的数据，当高优先级数据缺失时，则依次使用备用方案。

整个逻辑主要在 `darktable/src/iop/temperature.c` 文件中的 `_find_coeffs` 函数中实现。

#### 1. 首选方案：直接读取 RAW 文件元数据

这是最直接、最准确的 “as shot” 实现方式。

*   **实现逻辑**: 程序首先尝试从图像的元数据中直接读取由相机在拍摄时记录的白平衡乘数。
*   **对应源码**: 在 `darktable/src/iop/temperature.c` 的 `_find_coeffs` 函数中，代码会检查 `img->wb_coeffs` 数组。
    ```c
    // a snippet from _find_coeffs in src/iop/temperature.c

    // the raw should provide wb coeffs:
    gboolean ok = TRUE;
    // ...
    for(int k = 0; ok && k < num_coeffs; k++)
    {
      if(!dt_isnormal(img->wb_coeffs[k]) || img->wb_coeffs[k] == 0.0f)
        ok = FALSE;
    }
    if(ok)
    {
      for_four_channels(k)
        coeffs[k] = img->wb_coeffs[k];
      return; // 成功读取，直接返回
    }
    ```
*   **说明**: `img->wb_coeffs` 存储了从 RAW 文件 EXIF 中解析出的白平衡系数。如果这些系数值有效（非 `NaN` 且不为零），系统会直接使用它们，流程结束。

#### 2. 备用方案 A：基于相机色彩矩阵计算

当 RAW 文件中没有有效的白平衡系数时，darktable 会尝试利用相机自身的色彩特性来计算一个标准的日光白平衡。

*   **实现逻辑**: 调用 `_calculate_bogus_daylight_wb` 函数，该函数使用存储在 `self->dev->image_storage` 中的相机输入色彩矩阵（如 `adobe_XYZ_to_CAM` 或 `d65_color_matrix`）来推算出一组在 D65 日光标准光源下的白平衡系数。
*   **对应源码**: 在 `darktable/src/iop/temperature.c` 中：
    ```c
    // a snippet from _find_coeffs in src/iop/temperature.c

    double bwb[4];
    if(!_calculate_bogus_daylight_wb(self, bwb))
    {
      // found camera matrix and used it to calculate bogus daylight wb
      for_four_channels(c)
        coeffs[c] = bwb[c];
      return; // 计算成功，返回
    }
    ```

#### 3. 备用方案 B：查询白平衡预设数据库

如果相机色彩矩阵缺失或无法用于计算，系统会查询内置的白平衡预设数据库。

*   **实现逻辑**: 系统会遍历一个庞大的预设数据库，该数据库包含了大量主流相机型号在特定光照条件下的标准白平衡系数。它会根据当前图像的相机制造商 (`img->camera_maker`) 和型号 (`img->camera_model`) 来查找匹配的条目。
*   **对应源码**: 在 `darktable/src/iop/temperature.c` 中：
    ```c
    // a snippet from _find_coeffs in src/iop/temperature.c

    // no cam matrix??? try presets:
    for(int i = 0; i < dt_wb_presets_count(); i++)
    {
      const dt_wb_data *wbp = dt_wb_preset(i);

      if(!strcmp(wbp->make, img->camera_maker)
         && !strcmp(wbp->model, img->camera_model))
      {
        // just take the first preset we find for this camera
        for(int k = 0; k < 3; k++)
          coeffs[k] = wbp->channels[k];
        return; // 找到预设，返回
      }
    }
    ```

#### 4. 最终保底方案：硬编码的默认值

在极少数情况下，如果以上所有方法都失败了（例如，一个非常罕见或全新的相机型号），darktable 为了保证程序的稳定运行，会使用一组硬编码的默认值。

*   **实现逻辑**: 在报告“无法读取相机白平衡信息”的错误后，提供一组适用于大多数相机的通用系数值。
*   **对应源码**: 在 `darktable/src/iop/temperature.c` 中：
    ```c
    // a snippet from _find_coeffs in src/iop/temperature.c

    // ...
    // final security net: hardcoded default that fits most cams.
    coeffs[0] = 2.0;
    coeffs[1] = 1.0;
    coeffs[2] = 1.5;
    coeffs[3] = 1.0;
    ``` 