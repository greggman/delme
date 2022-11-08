import React, { CSSProperties } from 'react';
import JsonValue, { JsonValueArrayValueArray } from '../JsonValue/JsonValue';

import './ColorPickerResult.css';

interface Props {
    position: { x: number; y: number };
    values: Float32Array;
    style: CSSProperties;
}

const ColorPickerResult: React.FC<Props> = ({ position, values, style = {} }: Props) => {
    return (
        <div className="spector2-colorpickerresult" style={style}>
            <table>
                <tr>
                    <td
                        rowSpan={2}
                        style={{
                            width: '40px',
                            backgroundColor: `rgb(${values[0] * 255}, ${values[1] * 255}, ${values[2] * 255})`,
                        }}
                    ></td>
                    <td>Coord:</td>
                    <td>
                        <JsonValueArrayValueArray data={[position.x, position.y]} />
                    </td>
                </tr>
                <tr>
                    <td>Values:</td>
                    <td>
                        <JsonValueArrayValueArray data={values} />
                    </td>
                </tr>
            </table>
        </div>
    );
};

export default ColorPickerResult;
